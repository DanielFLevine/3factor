# Based on the code for the Stimulus-response task as described in Miconi et al. ICLR 2019 and Miconi and Kay Neuron 2025.

import argparse
import logging
import torch
import numpy as np
from numpy import random
import torch.nn.functional as F
import random
import time
import platform
from pathlib import Path
from datetime import datetime

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

import numpy as np
import constants as constants_module

from rnn import RetroModulRNN
from params import params
from constants import (
    DEVICE, ADDINPUT, NUMRESPONSESTEP,
    PROBAOLDDATA, POSALPHA, EVAL, RESETHIDDENEVERYTRIAL, RESETETEVERYTRIAL,
    RESETPWEVERYTRIAL, SECONDMODULATOR, TURN_OFF_RECURRENT_WEIGHTS, ZERO_RECURRENT_WEIGHTS,
    get_nbstimbits, get_input_size
)

from rng_helpers import capture_rng_state, restore_rng_state
from utils import get_run_metadata, RunMetadataBundle, save_checkpoint, dump_json, serialize_for_json, snapshot_configuration, get_constants_dict, write_metrics_csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
wandb_run = None


parser = argparse.ArgumentParser(description="Train MetaTI with optional checkpointing")
parser.add_argument('--seed', type=int, default=-1, help="Random seed (ignored when resuming)")
parser.add_argument('--resume-dir', type=str, default=None, help="Path to an existing run directory to resume from")
parser.add_argument('--checkpoint-path', type=str, default=None, help="Override checkpoint path (defaults to <run_dir>/checkpoint.pt)")
parser.add_argument('--neuromodulator-count', type=int, default=1, help="Number of neuromodulators to use")
parser.add_argument('--hidden-size', type=int, default=200, help="Size of the hidden layer")
parser.add_argument('--input-plastic', action='store_true', help="Use plastic weights for input connections")
parser.add_argument('--fixed-random-plastic-init', action='store_true', help="Use fixed random initialization for plastic weights (resets to same values each episode)")
parser.add_argument('--wandb-project', type=str, default=None, help="Weights & Biases project name")
parser.add_argument('--wandb-entity', type=str, default=None, help="Weights & Biases entity (team) name")
parser.add_argument('--wandb-mode', type=str, default=None, help="Weights & Biases mode (online, offline, disabled)")
parser.add_argument('--wandb-run-id', type=str, default=None, help="Explicit Weights & Biases run ID to resume")
parser.add_argument('--wandb-name', type=str, default=None, help="Explicit Weights & Biases run display name")
parser.add_argument('--wandb-tags', type=str, nargs='*', default=None, help="Tags to attach to the Weights & Biases run")
parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging")
args = parser.parse_args()

resume_dir = Path(args.resume_dir).resolve() if args.resume_dir else None
is_resuming = resume_dir is not None
checkpoint_state = None

run_metadata_bundle = get_run_metadata(args)

run_metadata, run_metadata_path, metrics_csv_path, checkpoint_path, metrics_rows, run_dir, params_path, constants_path, seed = run_metadata_bundle.as_tuple()

run_metadata["metrics_file"] = str(metrics_csv_path)
run_metadata["checkpoint_file"] = str(checkpoint_path)
run_metadata["metrics_rows"] = len(metrics_rows)
dump_json(run_metadata, run_metadata_path)


def wandb_log(data, *, step=None, commit=True):
    if wandb_run is None or not data:
        return
    sanitized = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, Path):
            value = str(value)
        elif isinstance(value, dict) or isinstance(value, (list, tuple, set)):
            value = serialize_for_json(value)
        # Allow wandb special types (Histogram, Image, etc.) to pass through
        if isinstance(value, (int, float, bool, str)) or (wandb is not None and isinstance(value, (wandb.Histogram, wandb.Image))):
            sanitized[key] = value
    if sanitized:
        wandb_run.log(sanitized, step=step, commit=commit)


# Update params with seed from command line
params['rngseed'] = seed
params['hidden_size'] = args.hidden_size
params['input_plastic'] = args.input_plastic
params['fixed_random_plastic_init'] = args.fixed_random_plastic_init
params['neuromodulator_count'] = args.neuromodulator_count

# Compute derived parameters
params['nbtrials'] = params['nbtraintrials'] +  params['nbtesttrials']
params['eplen'] = params['nbtrials'] * params['triallen']

np.set_printoptions(precision=5)
device = DEVICE



logger.info("Starting...")

logger.info("Passed params: %s", params)
logger.info(platform.uname())
suffix = "_"+"".join( [str(kk)+str(vv)+"_" if kk != 'pe' and kk != 'nbsteps' and kk != 'rngseed' and kk != 'save_every' and kk != 'test_every' else '' for kk, vv in sorted(zip(params.keys(), params.values()))] ) + "_rng" + str(params['rngseed'])  # Turning the parameters into a nice suffix for filenames
logger.info(suffix)
run_metadata["suffix"] = suffix
dump_json(run_metadata, run_metadata_path)


# Total input size = cue size +  one 'go' bit + 4 additional inputs
NBSTIMBITS = get_nbstimbits(params['cue_size'])
params['outputsize'] = 2  # "response" and "no response"
params['inputsize'] = get_input_size(params['cue_size'], params['outputsize'])

wandb_enabled = False
if args.no_wandb:
    logger.info("Weights & Biases logging disabled via --no-wandb flag")
    run_metadata["wandb_enabled"] = False
    dump_json(run_metadata, run_metadata_path)
elif wandb is None:
    logger.warning("Weights & Biases is not installed; skipping wandb logging")
    run_metadata["wandb_enabled"] = False
    dump_json(run_metadata, run_metadata_path)
else:
    wandb_project = args.wandb_project or run_metadata.get("wandb_project")
    wandb_entity = args.wandb_entity or run_metadata.get("wandb_entity")
    wandb_mode = args.wandb_mode or run_metadata.get("wandb_mode")
    wandb_run_id = args.wandb_run_id or run_metadata.get("wandb_run_id")
    wandb_name = args.wandb_name or run_metadata.get("wandb_name")
    wandb_tags = args.wandb_tags if args.wandb_tags is not None else run_metadata.get("wandb_tags")

    if wandb_run_id is None:
        wandb_run_id = wandb.util.generate_id()

    if isinstance(wandb_tags, str):
        wandb_tags = [wandb_tags]

    wandb_config = {
        "params": {k: serialize_for_json(v) for k, v in params.items()},
        "constants": get_constants_dict(),
        "is_resuming": is_resuming,
        "run_directory": str(run_dir),
    }

    init_kwargs = {
        "config": wandb_config,
        "dir": str(run_dir),
        "id": wandb_run_id,
        "resume": "allow",
    }
    if wandb_project:
        init_kwargs["project"] = wandb_project
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_name:
        init_kwargs["name"] = wandb_name
    if wandb_tags:
        init_kwargs["tags"] = list(wandb_tags)
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode

    try:
        wandb_run = wandb.init(**init_kwargs)
        wandb_enabled = wandb_run is not None
    except Exception as exc:  # pragma: no cover - initialization failures
        wandb_run = None
        wandb_enabled = False
        logger.warning("Failed to initialize Weights & Biases run: %s", exc)

    if wandb_enabled:
        run_metadata.update({
            "wandb_enabled": True,
            "wandb_project": wandb_run.project,
            "wandb_entity": wandb_run.entity,
            "wandb_mode": wandb_mode,
            "wandb_run_id": wandb_run.id,
            "wandb_name": wandb_run.name,
            "wandb_tags": list(wandb_run.tags) if wandb_run.tags else None,
            "wandb_url": wandb_run.url,
        })
        dump_json(run_metadata, run_metadata_path)
    else:
        run_metadata["wandb_enabled"] = False
        dump_json(run_metadata, run_metadata_path)


# Initialize random seeds, unless rngseed is -1 (first two redundant?)
if params['rngseed'] > -1 :
    logger.info("Setting random seed %s", params['rngseed'])
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
else:
    logger.info("No random seed.")



# All experimental flags and constants are now imported from constants.py


batch_size = params['batch_size']   # Batch size
if not is_resuming:
    snapshot_configuration(run_metadata, params_path, constants_path, run_metadata_path)

logger.info("Number of neuromodulators: %s", args.neuromodulator_count)
params['neuromodulator_count'] = args.neuromodulator_count

logger.info("Initializing network")
net = RetroModulRNN(params)
wandb.watch(net, log="all", log_freq=100)


logger.info("Shape of all optimized parameters: %s", [x.size() for x in net.parameters()])
allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
logger.info("Size (numel) of all optimized elements: %s", allsizes)
logger.info("Total size (numel) of all optimized elements: %s", sum(allsizes))

logger.info("Initializing optimizer")
optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=params['eps'], weight_decay=params['l2'])

# Training state (supports resume) ------------------------------------------
start_episode = 0
all_losses = []
all_grad_norms = []
all_losses_objective = []
all_mean_rewards_ep = []
all_mean_testrewards_ep = []
all_losses_v = []

old_cue_data = []
lossbetweensaves = 0.0
nowtime = time.time()
totalnbtrials = 0
nbtrialswithcc = 0

if checkpoint_state:
    model_state = checkpoint_state.get('model_state_dict')
    if model_state:
        net.load_state_dict(model_state)
    optimizer_state = checkpoint_state.get('optimizer_state_dict')
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    all_losses_objective = [float(x) for x in checkpoint_state.get('all_losses_objective', all_losses_objective)]
    all_mean_rewards_ep = [float(x) for x in checkpoint_state.get('all_mean_rewards_ep', all_mean_rewards_ep)]
    all_mean_testrewards_ep = [float(x) for x in checkpoint_state.get('all_mean_testrewards_ep', all_mean_testrewards_ep)]
    all_grad_norms = [float(x) for x in checkpoint_state.get('all_grad_norms', all_grad_norms)]
    lossbetweensaves = float(checkpoint_state.get('lossbetweensaves', lossbetweensaves))
    old_cue_data = checkpoint_state.get('old_cue_data', old_cue_data)
    totalnbtrials = int(checkpoint_state.get('totalnbtrials', totalnbtrials))
    nbtrialswithcc = int(checkpoint_state.get('nbtrialswithcc', nbtrialswithcc))
    restored_nowtime = checkpoint_state.get('nowtime')
    if restored_nowtime is not None:
        nowtime = float(restored_nowtime)
    start_episode = int(checkpoint_state.get('next_episode', 0))
    rng_state = checkpoint_state.get('rng_state')
    if rng_state is not None:
        restore_rng_state(rng_state)
    run_metadata["resumed_at_episode"] = start_episode
    run_metadata["resume_checkpoint_created_at"] = checkpoint_state.get('checkpoint_created_at')
    dump_json(run_metadata, run_metadata_path)
    logger.info("Checkpoint restored from %s; resuming at episode %s", checkpoint_path, start_episode)

nbtrials = [0] * batch_size

logger.info("Starting episodes!")


for numepisode in range(start_episode, params['number_of_training_iterations']):


    PRINTTRACE = False
    if (numepisode) % (params['pe']) == 0 or EVAL:
        PRINTTRACE = True


    optimizer.zero_grad()
    loss = 0
    lossv = 0
    lossnms = [torch.tensor(0.0, device=device) for _ in range(params['neuromodulator_count'])]
    lossHL1 = 0
    # The freshly generated cue data will be appended to old cue data later, after the episode is run
    if numepisode % params['number_of_episodes_between_resets'] == 0:
        params['number_of_cues'] = random.choice(params['number_of_cues_range'])
        old_cue_data = []
        hidden = net.initialZeroState(batch_size)
        eligibility_trace = net.initialZeroET(batch_size) #  The Hebbian eligibility trace
        plastic_weights = [net.initialZeroPlasticWeights(batch_size, nm_idx=i) for i in range(params['neuromodulator_count'])]
        if params.get('input_plastic', False):
            input_eligibility_trace = net.initialZeroInputET(batch_size)
            input_plastic_weights = net.initialZeroInputPlasticWeights(batch_size)

        if params['reset_random_recurrent_init'] and TURN_OFF_RECURRENT_WEIGHTS and (not ZERO_RECURRENT_WEIGHTS):
            logger.info("Resetting recurrent weights to random values")
            net.w = w = torch.nn.Parameter((  (1.0 / np.sqrt(params['hidden_size']))  * ( 2.0 * torch.rand(params['hidden_size'], params['hidden_size']) - 1.0) ).to(device), requires_grad=False)
    else:
        hidden = hidden.detach()
        eligibility_trace = eligibility_trace.detach()
        plastic_weights = [plastic_weight.detach() for plastic_weight in plastic_weights]
        if params.get('input_plastic', False):
            input_eligibility_trace = input_eligibility_trace.detach()
            input_plastic_weights = input_plastic_weights.detach()

    numstep_ep = 0
    iscorrect_thisep = np.zeros((batch_size, params['nbtrials']))
    istest_thisep  = np.zeros((batch_size, params['nbtrials']))
    isadjacent_thisep  = np.zeros((batch_size, params['nbtrials']))
    isolddata_thisep  = np.zeros((batch_size, params['nbtrials']))
    resps_thisep =  np.zeros((batch_size, params['nbtrials']))
    cuepairs_thisep  = []
    numactionschosen_alltrialsandsteps_thisep = np.zeros((batch_size, params['nbtrials'], params['triallen'])).astype(int)


    # Generate the bitstring for each cue number for this episode. Make sure they're all different (important when using very small cues for debugging, e.g. cue_size=2, number_of_cues>4)
    cue_data_for_this_episode =[]
    for batch_index in range(batch_size):
        cue_data_for_this_episode.append([])
        for cue_number in range(params['number_of_cues']):
            assert len(cue_data_for_this_episode[batch_index]) == cue_number # Sanity check to make sure we're not overwriting existing cue data and we're adding the correct number of cues
            found_same_cue = True
            counter_for_this_cue = 0 # Counter to prevent infinite loop in case we can't find a different cue. Occurs when small number of cues and cue_size are used.
            while found_same_cue:
                counter_for_this_cue += 1
                if counter_for_this_cue > 10000:
                    # This should only occur with very weird parameters, e.g. cue_size=2, number_of_cues>4
                    raise ValueError("Could not generate a full list of different cues")
                found_same_cue = False
                candidate = np.random.randint(2, size=params['cue_size']) * 2 - 1
                for backtrace in range(cue_number): # Candidate cue must be at least 34% different from all previous cues in the episode
                    if np.mean(cue_data_for_this_episode[batch_index][backtrace] == candidate) > .66 :
                        found_same_cue = True

            cue_data_for_this_episode[batch_index].append(candidate)


    reward = np.zeros(batch_size)
    sumreward = np.zeros(batch_size)
    sumrewardtest = np.zeros(batch_size)
    rewards = []
    vs = []
    logprobs = []
    cues=[]
    for nb in range(batch_size):
        cues.append([])
    dist = 0
    numactionschosen = np.zeros(batch_size, dtype='int32')

    nbtrials = np.zeros(batch_size)
    nbtesttrials = nbtesttrials_correct = nbtesttrials_adjcues = nbtesttrials_adjcues_correct = nbtesttrials_nonadjcues = nbtesttrials_nonadjcues_correct = 0
    nbrewardabletrials = np.zeros(batch_size)
    thistrialhascorrectorder = np.zeros(batch_size)
    thistrialhasadjacentcues = np.zeros(batch_size)
    thistrialhascorrectanswer = np.zeros(batch_size)


    # 2 steps of blank input between episodes. Not sure if it helps.
    inputs = np.zeros((batch_size, params['inputsize']), dtype='float32')
    inputsC = torch.from_numpy(inputs).detach().to(device)
    for nn in range(2):
        if params.get('input_plastic', False):
            y, v, nmscalars, hidden, eligibility_trace, plastic_weights, input_eligibility_trace, input_plastic_weights = net(inputsC, hidden, eligibility_trace, plastic_weights, input_eligibility_trace, input_plastic_weights)
        else:
            y, v, nmscalars, hidden, eligibility_trace, plastic_weights  = net(inputsC, hidden, eligibility_trace, plastic_weights)  # y  should output raw scores, not probas


    for numtrial  in  range(params['nbtrials']):
        if RESETHIDDENEVERYTRIAL:
            hidden = net.initialZeroState(batch_size)
        if RESETETEVERYTRIAL:
            # et = et * 0 # net.initialZeroET()
            eligibility_trace = net.initialZeroET(batch_size)
            if params.get('input_plastic', False):
                input_eligibility_trace = net.initialZeroInputET(batch_size)
        if RESETPWEVERYTRIAL:
            plastic_weights = [net.initialZeroPlasticWeights(batch_size, nm_idx=i) for i in range(params['neuromodulator_count'])]
            if params.get('input_plastic', False):
                input_plastic_weights = net.initialZeroInputPlasticWeights(batch_size)

        hiddens0=[]

        cuepairs_thistrial = []
        for nb in range(batch_size):
            thistrialhascorrectorder[nb] = 0
            cuerange = range(params['number_of_cues'])

            # # In any trial, we show exactly two cues (randomly chosen), simultaneously:
            cuepair =  list(np.random.choice(cuerange, 2, replace=False))

            # If the trial is NOT a test trial, these two cues should be adjacent
            if nbtrials[nb]  < params['nbtraintrials']:
                while abs(cuepair[0] - cuepair[1]) > 1 :
                    cuepair =  list(np.random.choice(cuerange, 2, replace=False))

            thistrialhascorrectorder[nb] = 1 if cuepair[0]  <  cuepair[1] else 0
            thistrialhasadjacentcues[nb] = 1 if (abs(cuepair[0]-cuepair[1]) == 1) else  0
            isadjacent_thisep[nb,numtrial]  = thistrialhasadjacentcues[nb]
            istest_thisep[nb, numtrial] = 1 if numtrial >= params['nbtraintrials'] else 0

            # mycues = [cuepair,cuepair]
            mycues = [cuepair,]
            cuepairs_thistrial.append(cuepair)

            mycues.append(params['number_of_cues']) # The 'go' cue, instructing response from the network
            mycues.append(-1) # One empty  step.During the first empty step, reward (computed on the previous step) is seen by the network.
            mycues.append(-1)
            assert len(mycues) == params['triallen']
            assert  mycues[NUMRESPONSESTEP] == params['number_of_cues']  # The 'response' step is signalled by the 'go' cue, whose number is params['number_of_cues'].
            cues[nb] = mycues

        cuepairs_thisep.append(cuepairs_thistrial)

        # In test period, if there ars some old cues in the store, some trials will use old cues
        if len(old_cue_data) > 0 and numtrial >= params['nbtraintrials']:
            for nb in range(batch_size):
                if np.random.rand() < PROBAOLDDATA:
                    isolddata_thisep[nb,numtrial] = 1



        # Now we are ready to actually  run  the trial:

        for numstep in range(params['triallen']):

            inputs = np.zeros((batch_size, params['inputsize']), dtype='float32')

            for nb in range(batch_size):
                # Turning the cue number for this time step into actual (signed) bitstring inputs, using the cue  data generated at the beginning of the episode - or, ocasionally, old_cue_data
                inputs[nb, :NBSTIMBITS] = 0
                if cues[nb][numstep] != -1 and cues[nb][numstep] != params['number_of_cues']:
                    assert len(cues[nb][numstep]) == 2
                    if isolddata_thisep[nb, numtrial]:
                        oldpos  = np.random.randint(len(old_cue_data))
                        inputs[nb, :NBSTIMBITS-1] = np.concatenate( ( old_cue_data[oldpos][nb][cues[nb][numstep][0]][:], old_cue_data[oldpos][nb][cues[nb][numstep][1]][:]  ) )
                    else:
                        inputs[nb, :NBSTIMBITS-1] = np.concatenate( ( cue_data_for_this_episode[nb][cues[nb][numstep][0]][:], cue_data_for_this_episode[nb][cues[nb][numstep][1]][:]  ) )

                    inputs0thistrial = inputs[0, :NBSTIMBITS-1]

                if cues[nb][numstep] == params['number_of_cues']:
                    inputs[nb, NBSTIMBITS-1] = 1  # "Go" cue

                inputs[nb, NBSTIMBITS + 0] = 1.0 # Bias neuron, probably not necessary
                inputs[nb,NBSTIMBITS +  1] = numstep_ep / params['eplen'] # Time passed in this episode. Should it be the trial? Doesn't matter much anyway.
                inputs[nb, NBSTIMBITS + 2] = 1.0 * reward[nb] # Reward from previous time step


                # Original:
                # if numstep > 0:
                #     inputs[nb, NBSTIMBITS + ADDINPUT + numactionschosen[nb]] = 1  # Previously chosen action
                # DEBUGGING !!
                # if numstep == 2:
                # if not (numtrial == 0 and numstep == 0):
                # if numstep == 0 and numtrial > 0:
                assert NUMRESPONSESTEP + 1 < params['triallen'] # If that is not the case, we must provide the action signal in the next trial (this works)
                if numstep == NUMRESPONSESTEP + 1:
                    inputs[nb, NBSTIMBITS + ADDINPUT + numactionschosen[nb]] = 1  # Previously chosen action


            # inputsC = torch.from_numpy(inputs, requires_grad=False).to(device)
            inputsC = torch.from_numpy(inputs).detach().to(device)


            ## Running the network
            if params.get('input_plastic', False):
                y, v, nmscalars, hidden, eligibility_trace, plastic_weights, input_eligibility_trace, input_plastic_weights = net(inputsC, hidden, eligibility_trace, plastic_weights, input_eligibility_trace, input_plastic_weights)
            else:
                y, v, nmscalars, hidden, eligibility_trace, plastic_weights  = net(inputsC, hidden, eligibility_trace, plastic_weights)  # y  should output raw scores, not probas

            hiddens0.append(hidden[0,:])





            # Choosing the action from the outputs
            y = F.softmax(y, dim=1)
            # Must convert y to probas to use this !
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()
            logprobs.append(distrib.log_prob(actionschosen))    # To be used later for the A2C algorithm
            # if numstep == NUMRESPONSESTEP: # 2: # 4: #3: #  2:
            #     logprobs.append(distrib.log_prob(actionschosen))    # To be used later for the A2C algorithm
            # else:
            #     logprobs.append(0)
            numactionschosen = actionschosen.data.cpu().numpy()    # Store as scalars (for the whole batch)

            if PRINTTRACE:
                logger.info(
                    (
                        f"Tr {numtrial} Step {numstep}, Cue 1 (0): {inputs[0,:params['cue_size']]}, Cue 2 (0): {inputs[0,params['cue_size']:2*params['cue_size']]}, "
                        f"Other inputs: {inputs[0, 2*params['cue_size']:]}\n - Outputs(0): {y.data.cpu().numpy()[0,:]} - action chosen(0): {numactionschosen[0]}, "
                        f"TrialLen: {params['triallen']}, numstep: {numstep}, TTHCC(0): {thistrialhascorrectorder[0]}, TTHOC(0): {isolddata_thisep[0, numtrial]}, "
                        f"Reward (based on prev step): {reward[0]}, DAout1: {nmscalars[0][0].detach().item()}, DAout2: {nmscalars[0][1].detach().item()}, cues(0): {cues[0]}"
                    )
                )

            # Computing the rewards. This is done for each time step.
            reward = np.zeros(batch_size, dtype='float32')
            for nb in range(batch_size):
                if numactionschosen[nb] == 1:
                    # Small penalty for any non-rest action taken
                    # In practice, this would usually be 0
                    reward[nb]  -= params['wp']

                numactionschosen_alltrialsandsteps_thisep[nb, numtrial, numstep] = numactionschosen[nb]

                if numstep == NUMRESPONSESTEP: # 2: # 4: #3: #  2:
                    # This is the 'response' step of the trial (and we showed the response signal
                    assert cues[nb][numstep] == params['number_of_cues']
                    resps_thisep[nb, numtrial] = numactionschosen[nb] *2 - 1    # Store the response in this timestep as the response for the whole trial, for logging/analysis purposes
                    # We must deliver reward (which will be perceived by the agent at the next step), positive or negative, depending on response
                    thistrialhascorrectanswer[nb] = 1
                    if thistrialhascorrectorder[nb] and numactionschosen[nb] == 1:
                        reward[nb] += params['rew']
                    elif (not thistrialhascorrectorder[nb]) and numactionschosen[nb] == 0:
                        reward[nb] += params['rew']
                    else:
                        reward[nb] -= params['rew']
                        thistrialhascorrectanswer[nb] = 0
                    iscorrect_thisep[nb, numtrial] = thistrialhascorrectanswer[nb]

                    if ( cuepairs_thistrial[nb][0]  < cuepairs_thistrial[nb][1]  ) and numactionschosen[nb] == 1:
                        assert thistrialhascorrectanswer[nb]
                    if ( cuepairs_thistrial[nb][0]  > cuepairs_thistrial[nb][1]  ) and numactionschosen[nb] == 1:
                        assert not thistrialhascorrectanswer[nb]
                    if ( cuepairs_thistrial[nb][0]  < cuepairs_thistrial[nb][1]  ) and numactionschosen[nb] == 0:
                        assert not thistrialhascorrectanswer[nb]
                    if ( cuepairs_thistrial[nb][0]  > cuepairs_thistrial[nb][1]  ) and numactionschosen[nb] == 0:
                        assert thistrialhascorrectanswer[nb]




                if numstep == params['triallen'] - 1:
                    # This was the last step of the trial
                    nbtrials[nb] += 1
                    totalnbtrials += 1
                    if thistrialhascorrectorder[nb]:
                        nbtrialswithcc += 1



            rewards.append(reward)
            vs.append(v)
            sumreward += reward
            if numtrial >= params['nbtrials'] - params['nbtesttrials']:
                sumrewardtest += reward
            # lossDA +=  torch.sum(torch.abs(DAout))
            step_lossnms = [
                torch.sum(torch.abs(nmscalar / (1e-8 + net.neuromodulator_multipliers[i])))
                for i, nmscalar in enumerate(nmscalars)
            ]   # This is a hack to "remove" DAmult from the L1 penalty. Assumes nm multipliers never go < 0.
            lossnms = [prev + curr for prev, curr in zip(lossnms, step_lossnms)]
            lossHL1 += torch.mean(torch.abs(hidden))


            loss += (params['bent'] * y.pow(2).sum() / batch_size )   # In real A2c, this is an entropy incentive. Our original version of PyTorch did not have an entropy() function for Distribution, so we use sum-of-squares instead.

            numstep_ep  += 1


        # All steps done for this trial
        if numtrial >= params['nbtrials'] - params['nbtesttrials']:
            sumrewardtest += reward
            nbtesttrials += batch_size
            nbtesttrials_correct += np.sum(thistrialhascorrectanswer)
            nbtesttrials_adjcues += np.sum(thistrialhasadjacentcues)
            nbtesttrials_adjcues_correct += np.sum(thistrialhasadjacentcues * thistrialhascorrectanswer)
            nbtesttrials_nonadjcues += np.sum(1 - thistrialhasadjacentcues)
            nbtesttrials_nonadjcues_correct += np.sum((1-thistrialhasadjacentcues) * thistrialhascorrectanswer)


    # All trials done for this episode

    old_cue_data.append(cue_data_for_this_episode)
    if EVAL:
        ds_thisep = np.hstack(ds_thisep)
        rs_thisep = np.hstack(rs_thisep)

    # Computing the various losses for A2C (outer-loop training)

    R = torch.zeros(batch_size, requires_grad=False).to(device)
    gammaR = params['gr']
    for numstepb in reversed(range(params['eplen'])) :
        R = gammaR * R + torch.from_numpy(rewards[numstepb]).detach().to(device)

        ctrR = R - vs[numstepb][:,0] # I think this is right...
        lossv += ctrR.pow(2).sum() / batch_size
        LOSSMULT  = params['testlmult'] if numstepb > params['eplen']  - params['triallen']  * params['nbtesttrials'] else 1.0

        # NOTE: We accumulate the logprobs from all time steps, even when the output is ignored (it is only used to sample response at time step 1, i.e. RESPONSETIME)
        # Unsurprisingly, performance is better if we anly record the logprobs for response time (and set them to 0 otherwise), but we keep this version because it was used in the paper.
        loss -= LOSSMULT * (logprobs[numstepb] * ctrR.detach()).sum() / batch_size  # Action poliy loss



    lossobj = float(loss)
    loss += params['blossv'] * lossv   # lossmult is not applied to value-prediction loss; is it right?...
    lossnms_total = sum(lossnms, torch.tensor(0.0, device=device))
    loss += params['lda'] * lossnms_total  # lossnms is loss on absolute value of nm scalars (see above)
    loss += params['lhl1']  * lossHL1
    loss /= params['eplen']
    losspws = [torch.mean(plastic_weight ** 2) * params['lpw'] for plastic_weight in plastic_weights]   # loss on squared final plastic weights is not divided by episode length
    loss += sum(losspws)
    if params.get('input_plastic', False):
        loss_input_pw = torch.mean(input_plastic_weights ** 2) * params['lpw']  # loss on squared final input plastic weights
        loss += loss_input_pw

    test_perf_overall_value = None
    test_perf_adjacent_value = None
    test_perf_nonadjacent_value = None
    test_perf_old_cues_value = None

    if PRINTTRACE:
        logger.info(
            (
                f"lossobj (with coeff): {lossobj / params['eplen']}, lossv (with coeff): {params['blossv'] * float(lossv) / params['eplen']}, "
            f"lossnms: {float(lossnms_total)}, lossHL1: {params['lhl1'] * float(lossHL1) / params['eplen']}, losspws: {sum(losspws)}"
            )
        )
        logger.info("Total reward for this episode(0): %s, Prop. of trials w/ rewarded cue: %s, Total Nb of trials: %s",
                   sumreward[0], (nbtrialswithcc / totalnbtrials), totalnbtrials)
        logger.info("Nb Test Trials: %s, Nb Test Trials AdjCues: %s, Nb Test Trials NonAdjCues: %s",
                   nbtesttrials, nbtesttrials_adjcues, nbtesttrials_nonadjcues)
        if nbtesttrials > 0:
            test_perf_overall_value = float(nbtesttrials_correct / nbtesttrials)
            test_perf_adjacent_value = float(nbtesttrials_adjcues_correct / nbtesttrials_adjcues) if nbtesttrials_adjcues > 0 else None
            test_perf_nonadjacent_value = float(nbtesttrials_nonadjcues_correct / nbtesttrials_nonadjcues) if nbtesttrials_nonadjcues > 0 else None
            old_cues_denom = np.sum(istest_thisep * isolddata_thisep)
            test_perf_old_cues_value = float(np.sum(iscorrect_thisep * istest_thisep * isolddata_thisep) / old_cues_denom) if old_cues_denom > 0 else None
            # Should always be the  case except for LinkedListsEval
            logger.info("Test Perf (both methods): %s, Test Perf AdjCues: %s, Test Perf NonAdjCues: %s, Test perf old cues: %s",
                       np.array([nbtesttrials_correct / nbtesttrials, np.sum(iscorrect_thisep * istest_thisep) / np.sum(istest_thisep)]),
                       np.array([(nbtesttrials_adjcues_correct / nbtesttrials_adjcues)]) if nbtesttrials_adjcues > 0 else 'N/A',
                       np.array([nbtesttrials_nonadjcues_correct / nbtesttrials_nonadjcues]) if nbtesttrials_nonadjcues > 0 else 'N/A',
                       np.array([np.sum(iscorrect_thisep * istest_thisep * isolddata_thisep) /  np.sum(istest_thisep * isolddata_thisep)])  if np.sum(istest_thisep * isolddata_thisep) > 0 else "N/A")


    grad_norm_value = None
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), params['gc'])
    grad_norm_value = float(gn)
    all_grad_norms.append(grad_norm_value)
    if numepisode > 100:  # Burn-in period
        optimizer.step()
        if POSALPHA:
            for alpha in net.alphas:
                torch.clip_(alpha.data, min=0)


    lossnum = float(loss)
    lossbetweensaves += lossnum
    mean_reward_value = float(sumreward.mean())
    mean_test_reward_value = float(sumrewardtest.mean())
    all_losses_objective.append(lossnum)
    all_mean_rewards_ep.append(mean_reward_value)
    all_mean_testrewards_ep.append(mean_test_reward_value)

    if wandb_enabled:
        basic_metrics = {
            "episode": numepisode,
            "loss": lossnum,
            "mean_reward": mean_reward_value,
            "mean_test_reward": mean_test_reward_value,
            "learning_rate": optimizer.param_groups[0].get('lr', params['lr']) if optimizer.param_groups else params['lr'],
        }
        if grad_norm_value is not None:
            basic_metrics["grad_norm"] = grad_norm_value

        # Log individual loss components
        loss_components = {
            "loss/policy": lossobj / params['eplen'],
            "loss/value": params['blossv'] * float(lossv) / params['eplen'],
            "loss/neuromodulator_l1": params['lda'] * float(lossnms_total),
            "loss/hidden_l1": params['lhl1'] * float(lossHL1) / params['eplen'],
            "loss/plastic_weights_l2": float(sum(losspws)),
            "loss/total": lossnum,
        }
        if params.get('input_plastic', False):
            loss_components["loss/input_plastic_weights_l2"] = float(loss_input_pw)
        basic_metrics.update(loss_components)

        # Log plastic weight distributions as histograms
        for i, pw in enumerate(plastic_weights):
            pw_data = pw.detach().cpu().numpy().flatten()
            basic_metrics[f"plastic_weights/nm_{i}"] = wandb.Histogram(pw_data)
            basic_metrics[f"plastic_weights/nm_{i}_mean"] = float(np.mean(np.abs(pw_data)))
            basic_metrics[f"plastic_weights/nm_{i}_std"] = float(np.std(pw_data))
            basic_metrics[f"plastic_weights/nm_{i}_min"] = float(np.min(pw_data))
            basic_metrics[f"plastic_weights/nm_{i}_max"] = float(np.max(pw_data))

        if params.get('input_plastic', False):
            input_pw_data = input_plastic_weights.detach().cpu().numpy().flatten()
            basic_metrics["plastic_weights/input"] = wandb.Histogram(input_pw_data)
            basic_metrics["plastic_weights/input_mean"] = float(np.mean(np.abs(input_pw_data)))
            basic_metrics["plastic_weights/input_std"] = float(np.std(input_pw_data))

        # Log alpha and eta values
        for i, alpha in enumerate(net.alphas):
            alpha_data = alpha.detach().cpu().numpy().flatten()
            basic_metrics[f"alpha/nm_{i}"] = wandb.Histogram(alpha_data)
            basic_metrics[f"alpha/nm_{i}_mean"] = float(np.mean(np.abs(alpha_data)))
        basic_metrics["eta"] = float(net.etaet.detach().cpu().numpy())
        for i, nm_mult in enumerate(net.neuromodulator_multipliers):
            basic_metrics[f"neuromodulator_multiplier/nm_{i}"] = float(nm_mult.detach().cpu().numpy())

        wandb_log(basic_metrics, step=numepisode, commit=not PRINTTRACE)

    if PRINTTRACE:

        mean_loss_last_pe = lossbetweensaves / params['pe']
        mean_reward_last_pe = np.sum(all_mean_rewards_ep[-params['pe']:]) / params['pe']
        mean_test_reward_last_pe = np.sum(all_mean_testrewards_ep[-params['pe']:]) / params['pe']
        test_perf_overall = test_perf_overall_value
        test_perf_adjacent = test_perf_adjacent_value
        test_perf_nonadjacent = test_perf_nonadjacent_value
        test_perf_old_cues = test_perf_old_cues_value

        logger.info("Episode %s ====", numepisode)
        logger.info("Mean loss: %s", mean_loss_last_pe)
        lossbetweensaves = 0
        logger.info("Mean reward per episode (over whole batch and last %s episodes): %s",
                   params['pe'], mean_reward_last_pe)
        logger.info("Mean test-time reward per episode (over whole batch and last %s episodes): %s",
                   params['pe'], mean_test_reward_last_pe)
        previoustime = nowtime
        nowtime = time.time()
        time_spent_last_pe = nowtime - previoustime
        logger.info("Time spent on last %s iters: %s", params['pe'], time_spent_last_pe)

        # logger.info(" etaet: %s, DAmult: %s, mean-abs pw: %s", net.etaet.data.cpu().numpy(), net.DAmult.data.cpu().numpy(), np.mean(np.abs(pw.data.cpu().numpy())))
        logger.info((
            f"etaet: {net.etaet.data.cpu().numpy()}, neuromodulator_multipliers: {[float(neuromodulator_multiplier) for neuromodulator_multiplier in net.neuromodulator_multipliers]}, "
            f"mean-abs plastic_weights: {[np.mean(np.abs(plastic_weight.data.cpu().numpy())) for plastic_weight in plastic_weights]}"
        ))
        if params.get('input_plastic', False):
            logger.info((
                f"input_neuromodulator_multiplier: {float(net.input_neuromodulator_multiplier)}, "
                f"mean-abs input_plastic_weights: {np.mean(np.abs(input_plastic_weights.data.cpu().numpy()))}"
            ))
        logger.info("min/max/med-abs w, alpha, pw")
        logger.info("w: %s %s %s", float(torch.min(net.w)), float(torch.max(net.w)), float(torch.median(torch.abs(net.w))))
        logger.info("alpha: %s %s %s", float(torch.min(net.alphas[0])), float(torch.max(net.alphas[0])), float(torch.median(torch.abs(net.alphas[0]))))
        logger.info("pw: %s %s %s", float(torch.min(plastic_weights[0])), float(torch.max(plastic_weights[0])), float(torch.median(torch.abs(plastic_weights[0]))))
        if params.get('input_plastic', False):
            logger.info("input_alpha: %s %s %s", float(torch.min(net.input_alpha)), float(torch.max(net.input_alpha)), float(torch.median(torch.abs(net.input_alpha))))
            logger.info("input_pw: %s %s %s", float(torch.min(input_plastic_weights)), float(torch.max(input_plastic_weights)), float(torch.median(torch.abs(input_plastic_weights))))

        episode_metrics = {
            "episode": numepisode,
            "timestamp": datetime.now().isoformat(),
            "loss": lossnum,
            "mean_reward": mean_reward_value,
            "mean_test_reward": mean_test_reward_value,
            "mean_loss_last_pe": float(mean_loss_last_pe),
            "mean_reward_last_pe": float(mean_reward_last_pe),
            "mean_test_reward_last_pe": float(mean_test_reward_last_pe),
            "time_spent_last_pe": float(time_spent_last_pe),
            "test_perf_overall": test_perf_overall,
            "test_perf_adjacent": test_perf_adjacent,
            "test_perf_nonadjacent": test_perf_nonadjacent,
            "test_perf_old_cues": test_perf_old_cues,
        }
        metrics_rows.append(episode_metrics)
        write_metrics_csv(metrics_rows, run_metadata_bundle)
        if wandb_enabled:
            wandb_log(episode_metrics, step=numepisode, commit=True)
        save_checkpoint(
            run_metadata_bundle=run_metadata_bundle,
            next_episode=numepisode + 1,
            net=net,
            optimizer=optimizer,
            all_losses_objective=all_losses_objective,
            all_mean_rewards_ep=all_mean_rewards_ep,
            all_mean_testrewards_ep=all_mean_testrewards_ep,
            all_grad_norms=all_grad_norms,
            lossbetweensaves=lossbetweensaves,
            old_cue_data=old_cue_data,
            totalnbtrials=totalnbtrials,
            nbtrialswithcc=nbtrialswithcc,
            nowtime=nowtime,
        )

    if (numepisode) % params['save_every'] == 0 and numepisode  > 0:
        losslast100 = np.mean(all_losses_objective[-100:])
        logger.info("Average loss over the last 100 episodes: %s", losslast100)
        logger.info("Saving local files...")

        if numepisode > 0:
            # logger.info("Saving model parameters...")
            # torch.save(net.state_dict(), 'net_'+suffix+'.dat')
            torch.save(net.state_dict(), 'netAE'+str(params['rngseed'])+'.dat')
            torch.save(net.state_dict(), 'net.dat')

        with open('tAE'+str(params['rngseed'])+'.txt', 'w') as thefile:
            for item in all_mean_testrewards_ep[::10]:
                    thefile.write("%s\n" % item)


if params['number_of_training_iterations'] > 0:
    save_checkpoint(
        run_metadata_bundle=run_metadata_bundle,
        next_episode=params['number_of_training_iterations'],
        net=net,
        optimizer=optimizer,
        all_losses_objective=all_losses_objective,
        all_mean_rewards_ep=all_mean_rewards_ep,
        all_mean_testrewards_ep=all_mean_testrewards_ep,
        all_grad_norms=all_grad_norms,
        lossbetweensaves=lossbetweensaves,
        old_cue_data=old_cue_data,
        totalnbtrials=totalnbtrials,
        nbtrialswithcc=nbtrialswithcc,
        nowtime=nowtime,
    )


if wandb_run is not None:
    wandb_run.finish()
