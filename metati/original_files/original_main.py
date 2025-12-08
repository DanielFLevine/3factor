# Based on the code for the Stimulus-response task as described in Miconi et al. ICLR 2019.

import argparse
import pdb
import torch
import torch.nn as nn
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform

import numpy as np
import glob



myseed = -1


# If running this code on a cluster, uncomment the following, and pass a RNG seed as the --seed parameter on the command line
# parser  = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=-1)
# args = parser.parse_args()
# myseed =  args.seed



# This needs to be before parameter initialization
NBMASSEDTRIALS = 0
MASSEDPAIR = [3,4]




np.set_printoptions(precision=5)
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# device = 'cpu'


params={}
params['rngseed']=myseed       # RNG seed, or -1 for no seed
params['rew']=1.0   # reward amount
params['wp']=.0     # penalty for taking action 1 (not used here)
params['bent']=.1       #  entropy incentive (actually sum-of-squares)
params['blossv']=.1     # value prediction loss coefficient
params['gr']=.9         # Gamma for temporal reward discounting

params['hs']=200        # Size of the RNN's hidden layer
params['bs']=32         # Batch size
params['gc']=2.0    # Gradient clipping
params['eps']=1e-6  # A parameter for Adam
params['nbiter']= 30000 # 60000
params['save_every']=200
params['pe']= 101  #"print every"


params['nbcuesrange'] = range(4,9)
# params['nbcues']= 5 # 7     # number  of inputs - number of different stimuli used for each episode

params['cs']= 15  # 10     # Cue size -  number of binary elements in each cue vector (not including the 'go' bit and additional inputs, see below)

params['triallen'] = 4 # 4  #  5 # 5 + 1 #  4 + 1  # Each trial has: stimulus presentation, 'go' cue, then 3 empty trials.
NUMRESPONSESTEP = 1
params['nbtraintrials'] = 20 # 22  # 20 #  5  #  The first  nbtraintrials are the "train" trials. This  is included in nbtrials.
params['nbtesttrials'] =  10 # 2 #  12 # 10  #  The last nbtesttrials are the "test" trials. This  is included in nbtrials.
params['nbtrials'] = params['nbtraintrials']  +  NBMASSEDTRIALS + params['nbtesttrials'] #  20   # Number of trials per episode
params['eplen'] = params['nbtrials'] * params['triallen']  # eplen = episode length
params['testlmult'] =  3.0   # multiplier for the loss during the test trials
params['l2'] = 0 # 1e-5 # L2 penalty
params['lr'] = 1e-4
params['lpw'] =  1e-4  #  3    # plastic weight loss
params['lda'] = 0 # 1e-4 # 3e-5 # 1e-4 # 1e-5 # DA output penalty
params['lhl1'] =  0 # 3e-5
params['nbepsbwresets'] = 3 # 1

PROBAOLDDATA = .25
POSALPHA = False
POSALPHAINITONLY = False
VECTALPHA =  False
SCALARALPHA = False
assert not (SCALARALPHA and VECTALPHA)  # One or the other


# RNN with plastic connections and neuromodulation ("DA").
# Plasticity only in the recurrent connections, for now.

class RetroModulRNN(nn.Module):
    def __init__(self, params):
        super(RetroModulRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 2  # 2 DA neurons, we  take the difference  - see below
        self.params = params
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).to(device)
        self.w =  torch.nn.Parameter((  (1.0 / np.sqrt(params['hs']))  * ( 2.0 * torch.rand(params['hs'], params['hs']) - 1.0) ).to(device), requires_grad=True)
        #self.alpha =  torch.nn.Parameter((.1 * torch.ones(params['hs'], params['hs'])).to(device), requires_grad=True)
        # self.alpha =  torch.nn.Parameter((.01 * torch.ones(params['hs'], params['hs'])).to(device), requires_grad=True)
        if SCALARALPHA:
            self.alpha =  .01 * (2.0 * torch.rand(1, 1) -1.0).to(device)  # # A single scalar, so all connections share the same single plasticity coefficient (still trained)
        elif VECTALPHA:
            self.alpha =  .01 * (2.0 * torch.rand(params['hs'], 1) -1.0).to(device)  # A column vector, so each neuron has a single plasticity coefficient applied to all its input connections
        else:
            self.alpha =  .01 * (2.0 * torch.rand(params['hs'], params['hs']) -1.0).to(device)
        if POSALPHA or  POSALPHAINITONLY:
            self.alpha = torch.abs(self.alpha)
        self.alpha =  torch.nn.Parameter(self.alpha, requires_grad=True)
        # self.etaet = torch.nn.Parameter((.5 * torch.ones(1)).to(device), requires_grad=True)  # Everyone has the same etaet
        self.etaet = torch.nn.Parameter((.7 * torch.ones(1)).to(device), requires_grad=True)  # Everyone has the same etaet
        # self.DAmult = torch.nn.Parameter((1.0 * torch.ones(1)).to(device), requires_grad=True)  # Everyone has the same DAmult
        self.DAmult = torch.nn.Parameter((1.0 * torch.ones(1)).to(device), requires_grad=True)  # Everyone has the same DAmult
        # self.DAmult = .2
        self.h2DA = torch.nn.Linear(params['hs'], NBDA).to(device)      # DA output
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).to(device)  # Actual output
        self.h2v = torch.nn.Linear(params['hs'], 1).to(device)          # V prediction

    def forward(self, inputs, hidden, et, pw):
            BATCHSIZE = inputs.shape[0]  #  self.params['bs']
            HS = self.params['hs']
            assert pw.shape[0] == hidden.shape[0] == et.shape[0] == BATCHSIZE

            # Multiplying inputs (i.e. current hidden  values) by the total recurrent weights, w + alpha  * plastic_weights
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, pw)),
                            hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Output layer. Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)  # Value prediction

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1
            DAout2 = torch.tanh(self.h2DA(hactiv))
            DAout = self.DAmult * (DAout2[:,0] - DAout2[:,1])[:,None] # DA output is the difference between two tanh neurons - allows negative, positive and easy stable 0 output (by jamming both neurons to max or min)


            # Eligibility trace gets stamped into the plastic weights  - gated by DAout
            deltapw = DAout.view(BATCHSIZE,1,1) * et
            pw = pw + deltapw

            torch.clip_(pw, min=-50.0, max=50.0)



            # Updating the eligibility trace - Hebbbian update with a simple decay
            # NOTE: the decay is for the eligibility trace, NOT the plastic weights (which never decay during a lifetime, i.e. an episode)
            deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product; at this point 'hactiv' is the output and 'hidden' is the input  (i.e. ativities from previous time step)
            # deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) -  et * hactiv[:, :, None] ** 2  # Oja's rule  (...? anyway, doesn't ensure stability with tanh and arbitrary damult / etaet)
            # deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) -   hactiv.view(BATCHSIZE, HS, 1) * et  # Instar rule (?)

            deltaet = torch.tanh(deltaet)

            et = (1 - self.etaet) * et + self.etaet *  deltaet
            # et =  deltaet

            hidden = hactiv
            return activout, valueout, DAout, hidden, et, pw




    def initialZeroET(self, mybs):
        # return torch.zeros(self.params['bs'], self.params['hs'], self.params['hs'], requires_grad=False).to(device)
        return torch.zeros(mybs, self.params['hs'], self.params['hs'], requires_grad=False).to(device)

    def initialZeroPlasticWeights(self,  mybs):
        return torch.zeros(mybs, self.params['hs'], self.params['hs'] , requires_grad=False).to(device)
    def initialZeroState(self, mybs):
        return torch.zeros(mybs, self.params['hs'], requires_grad=False ).to(device)



print("Starting...")

print("Passed params: ", params)
print(platform.uname())
suffix = "_"+"".join( [str(kk)+str(vv)+"_" if kk != 'pe' and kk != 'nbsteps' and kk != 'rngseed' and kk != 'save_every' and kk != 'test_every' else '' for kk, vv in sorted(zip(params.keys(), params.values()))] ) + "_rng" + str(params['rngseed'])  # Turning the parameters into a nice suffix for filenames
print(suffix)


# Total input size = cue size +  one 'go' bit + 4 additional inputs
ADDINPUT = 4 # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs
NBSTIMBITS = 2 * params['cs'] + 1 # The additional bit is for the response cue (i.e. the "Go" cue)
params['outputsize'] =  2  # "response" and "no response"
params['inputsize'] = NBSTIMBITS  + ADDINPUT +  params['outputsize'] # The total number of input bits is the size of cues, plus the "response cue" binary input, plus the number of additional inputs, plus the number of actions


# Initialize random seeds, unless rngseed is -1 (first two redundant?)
if params['rngseed'] > -1 :
    print("Setting random seed", params['rngseed'])
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
else:
    print("No random seed.")





# Are we running in evaluation mode?
EVAL = False


# Various possible experiments:

RESETHIDDENEVERYTRIAL = RESETETEVERYTRIAL = True #  False #  True



RESETPWEVERYTRIAL = False


ONLYTWOLASTADJ = False

LINKEDLISTSEVAL = False
LINKINGISSHAM = False

FIXEDCUES = False

HALFNOBARREDPAIRUNTILT18 = False  # Ensures that  half the batch never sees the "barred" pair before trial 18. This should only be used for one thing:  ensuring enough selects and selectadd's when looking at single-step weight changes, so that some figures look better.
BARREDPAIR = [3,4]
#BARREDPAIR = [2,3]
#BARREDPAIR = [4,5]
# TO MAKE THE PLOTS WITH ADDITIONAL BARRED PAIR (for the "coupling" section):
#  1- Set the proper ADDBARREDPAIR below (the pair just before or just after the main BARRED PAIR)
#  2- Set SHOWALLSELECTS = False below



if EVAL:
    params['nbiter'] = 1 # 5 # 10
    params['bs']  = 2000
    params['nbcues'] = 8
    if not LINKEDLISTSEVAL:
        params['nbepsbwresets'] =  1
    torch.set_grad_enabled(False)
if LINKEDLISTSEVAL:
    assert EVAL
    assert NBMASSEDTRIALS==0
    assert params['nbepsbwresets'] == 3
    params['nbiter'] = 3
    params['nbcues'] = 8 # 10
    params['bs'] = 4000
    SHOWFIRSTHALFFIRST = 1 # np.random.randint(2)
    # The following applies for the first 2 episodes, then will be modified later for the 3rd episode
    params['nbtraintrials']  = 10
    params['nbtesttrials'] = 0
    params['nbtrials'] = params['nbtraintrials'] + params['nbtesttrials']
    params['eplen'] = params['nbtrials'] * params['triallen']  # eplen = episode length

if FIXEDCUES:
    params['bs']  =  2000

BS = params['bs']   # Batch size


assert not (  (NBMASSEDTRIALS > 0 ) and (not EVAL)  )   # We should only use massed trials in eval, not training
if ONLYTWOLASTADJ:
    assert params['nbcues'] == 7
if HALFNOBARREDPAIRUNTILT18:
    assert  EVAL and (NBMASSEDTRIALS == 0) and not LINKEDLISTSEVAL and not ONLYTWOLASTADJ and not FIXEDCUES
if LINKINGISSHAM:
    assert LINKEDLISTSEVAL




MIXNETWORKS = False


print("Initializing network")
net = RetroModulRNN(params)
if EVAL:
    net.load_state_dict(torch.load('net.dat'))
    net.eval()
    if MIXNETWORKS:
        netB = RetroModulRNN(params)
        netB.load_state_dict(torch.load('netB.dat'))
        # net.i2h = netB.i2h
        net.h2o = netB.h2o

    # net.alpha *= -1; net.DAmult *= -1   # Should leave the system invariant







print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
print ("Size (numel) of all optimized elements:", allsizes)
print ("Total size (numel) of all optimized elements:", sum(allsizes))

print("Initializing optimizer")
optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=params['eps'], weight_decay=params['l2'])

# A lot of logging...
all_losses = []
all_grad_norms = []
all_losses_objective = []
all_mean_rewards_ep = []
all_mean_testrewards_ep = []
all_losses_v = []

oldcuedata  = []

lossbetweensaves = 0
nowtime = time.time()

nbtrials = [0]*BS
totalnbtrials = 0
nbtrialswithcc = 0

print("Starting episodes!")


for numepisode in range(params['nbiter']):


    PRINTTRACE = False
    if (numepisode) % (params['pe']) == 0 or EVAL:
        PRINTTRACE = True



    if LINKEDLISTSEVAL and numepisode == 2:
        params['nbtraintrials']  = 1 if LINKINGISSHAM  else 4 #  12 #  7
        params['nbtesttrials'] = 1
        params['nbtrials'] = params['nbtraintrials'] + params['nbtesttrials']
        params['eplen'] = params['nbtrials'] * params['triallen']  # eplen = episode length

    optimizer.zero_grad()
    loss = 0
    lossv = 0
    lossDA = 0
    lossHL1 = 0
    # The freshly generated uedata will be appended to oldcuedata later, after the episode is run
    if numepisode % params['nbepsbwresets'] == 0:
        if not EVAL:
            params['nbcues']= random.choice(params['nbcuesrange'])
        oldcuedata = []
        hidden = net.initialZeroState(BS)
        et = net.initialZeroET(BS) #  The Hebbian eligibility trace
        pw = net.initialZeroPlasticWeights(BS)
    else:
        hidden = hidden.detach()
        et = et.detach()
        pw = pw.detach()

    numstep_ep = 0
    iscorrect_thisep = np.zeros((BS, params['nbtrials']))
    istest_thisep  = np.zeros((BS, params['nbtrials']))
    isadjacent_thisep  = np.zeros((BS, params['nbtrials']))
    isolddata_thisep  = np.zeros((BS, params['nbtrials']))
    resps_thisep =  np.zeros((BS, params['nbtrials']))
    cuepairs_thisep  = []
    numactionschosen_alltrialsandsteps_thisep = np.zeros((BS, params['nbtrials'], params['triallen'])).astype(int)
    if EVAL:
        allpwsavs_thisep = []
        ds_thisep =[]; rs_thisep  = []
        allrates_thisep = np.zeros((BS, params['hs'], params['eplen']))


    # Generate the bitstring for each cue number for this episode. Make sure they're all different (important when using very small cues for debugging, e.g. cs=2, ni=2)



    # print("Generating cues...")
    if FIXEDCUES:
        # Debugging only: Never change cue data
        if  numepisode == 0:
            cuedata=[]
            for nb in range(BS):
                cuedata.append([])
                for ncue in range(params['nbcues']):
                    if nb == 0:
                        assert len(cuedata[nb]) == ncue
                        foundsame = 1
                        cpt = 0
                        while foundsame > 0 :
                            cpt += 1
                            if cpt > 10000:
                                # This should only occur with very weird parameters, e.g. cs=2, ni>4
                                raise ValueError("Could not generate a full list of different cues")
                            foundsame = 0
                            candidate = np.random.randint(2, size=params['cs']) * 2 - 1
                            for backtrace in range(ncue):
                                # if np.array_equal(cuedata[nb][backtrace], candidate):
                                if np.mean(cuedata[nb][backtrace] == candidate) > .66 :
                                # if np.abs(np.mean(cuedata[nb][backtrace] * candidate)) > .1 :
                                    foundsame = 1
                        cuedata[nb].append(candidate)
                    else:
                        cuedata[nb].append(cuedata[0][ncue])

    else:  # Not fixed cues
        if not LINKEDLISTSEVAL or numepisode == 0:
        # if numepisode == 0:   # THIS DOESN't WORK TO FIX CUES! Different nb's still have different cues
            cuedata=[]
            for nb in range(BS):
                cuedata.append([])
                for ncue in range(params['nbcues']):
                    assert len(cuedata[nb]) == ncue
                    foundsame = 1
                    cpt = 0
                    while foundsame > 0 :
                        cpt += 1
                        if cpt > 10000:
                            # This should only occur with very weird parameters, e.g. cs=2, ni>4
                            raise ValueError("Could not generate a full list of different cues")
                        foundsame = 0
                        candidate = np.random.randint(2, size=params['cs']) * 2 - 1
                        for backtrace in range(ncue):
                            # if np.array_equal(cuedata[nb][backtrace], candidate):
                            # if np.abs(np.mean(cuedata[nb][backtrace] * candidate)) > .2 :
                            # if np.sum(cuedata[nb][backtrace] != candidate) < 4: # 2:
                            if np.mean(cuedata[nb][backtrace] == candidate) > .66 :
                                foundsame = 1

                    cuedata[nb].append(candidate)
    # print("Cues generated!")
    # print(len(cuedata), len(cuedata[0]), cuedata[0][0].shape)


    # One-hot encoded cues (though with random numbers for each batch element)
    if False:
        cuedata = []
        for nb in range(BS):
            xcues=[]
            order = np.arange(params['nbcues'])
            np.random.shuffle(order)
            for nc in range(params['nbcues']):
                cuevect = np.ones(params['cs']).astype(int) * -1
                cuevect[order[nc]] = 1
                xcues.append(cuevect)
            cuedata.append(xcues)
        # print(len(cuedata), len(cuedata[0]), cuedata[0][0].shape)



    # # The freshly generated cuedata will be appended to oldcuedata later, after the episode is run
    # if numepisode % params['nbepsbwresets'] == 0:
    #     oldcuedata = []

    reward = np.zeros(BS)
    sumreward = np.zeros(BS)
    sumrewardtest = np.zeros(BS)
    rewards = []
    vs = []
    logprobs = []
    cues=[]
    for nb in range(BS):
        cues.append([])
    dist = 0
    numactionschosen = np.zeros(BS, dtype='int32')

    #reward = 0.0
    #rewards = []
    #vs = []
    #logprobs = []
    #sumreward = 0.0
    nbtrials = np.zeros(BS)
    nbtesttrials = nbtesttrials_correct = nbtesttrials_adjcues = nbtesttrials_adjcues_correct = nbtesttrials_nonadjcues = nbtesttrials_nonadjcues_correct = 0
    nbrewardabletrials = np.zeros(BS)
    thistrialhascorrectorder = np.zeros(BS)
    thistrialhasadjacentcues = np.zeros(BS)
    thistrialhascorrectanswer = np.zeros(BS)


    # 2 steps of blank input between episodes. Not sure if it helps.
    inputs = np.zeros((BS, params['inputsize']), dtype='float32')
    inputsC = torch.from_numpy(inputs).detach().to(device)
    for nn in range(2):
            y, v, DAout, hidden, et, pw  = net(inputsC, hidden, et, pw)  # y  should output raw scores, not probas



    #print("EPISODE ", numepisode)

    for numtrial  in  range(params['nbtrials']):


        if RESETHIDDENEVERYTRIAL:
            hidden = net.initialZeroState(BS)
        if RESETETEVERYTRIAL:
            # et = et * 0 # net.initialZeroET()
            et = net.initialZeroET(BS)
        if RESETPWEVERYTRIAL:
            pw = net.initialZeroPlasticWeights(BS)

        hiddens0=[]

        # First, we prepare the specific sequence of inputs for this trial
        # The inputs can be a pair of cue numbers, or -1 (empty stimulus), or a single number equal to params['nbcues'], which indicates the 'response' cue.
        # These will be translated into actual network inputs (using the actual bitstrings) later.
        # Remember that the actual data for each cue  (i.e. its actual bitstring) is randomly generated for each episode, above

        cuepairs_thistrial = []
        for nb in range(BS):
                thistrialhascorrectorder[nb] = 0
                cuerange = range(params['nbcues'])
                if LINKEDLISTSEVAL:
                    if SHOWFIRSTHALFFIRST:
                        if numepisode == 0:
                            cuerange = range(params['nbcues']//2)
                        elif numepisode == 1:
                            cuerange  = range(params['nbcues']//2, params['nbcues'])
                        else:
                            cuerange =  range(params['nbcues'])
                    else:
                        if numepisode == 0:
                            cuerange  = range(params['nbcues']//2, params['nbcues'])
                        elif numepisode == 1:
                            cuerange = range(params['nbcues']//2)
                        else:
                            cuerange = range(params['nbcues'])
                # # In any trial, we show exactly two cues (randomly chosen), simultaneously:
                cuepair =  list(np.random.choice(cuerange, 2, replace=False))

                # If the trial is NOT a test trial, these two cues should be adjacent
                if nbtrials[nb]  < params['nbtraintrials'] or (ONLYTWOLASTADJ and nbtrials[nb] >= params['nbtrials'] - 2):
                    if ONLYTWOLASTADJ and nbtrials[nb] >= params['nbtrials'] - 2:
                        while abs(cuepair[0] - cuepair[1]) > 1 or 0 in cuepair or 6 in cuepair:
                            cuepair =  list(np.random.choice(cuerange, 2, replace=False))
                    else:
                            while abs(cuepair[0] - cuepair[1]) > 1 :
                                cuepair =  list(np.random.choice(cuerange, 2, replace=False))
                else:
                    assert nbtrials[nb] >= params['nbtraintrials']
                    if ONLYTWOLASTADJ:
                        assert nbtrials[nb] < params['nbtrials'] - 2
                        while  not(
                            (2  in cuepair and 0 in cuepair )
                            or (2  in cuepair and 4 in cuepair )
                            or (4  in cuepair and 6 in cuepair )
                            or (3  in cuepair and 0 in cuepair )
                            or (3  in cuepair and 6 in cuepair )
                        ):
                            cuepair =  list(np.random.choice(cuerange, 2, replace=False))

                if NBMASSEDTRIALS >  0 and nbtrials[nb]  >= params['nbtraintrials']  and numtrial < params['nbtrials'] -  params['nbtesttrials']:
                    cuepair  = MASSEDPAIR

                if LINKEDLISTSEVAL and numepisode  == 2:
                    if numtrial < params['nbtraintrials']:
                        if LINKINGISSHAM:
                            cuepair = [params['nbcues']//2-3,params['nbcues']//2-2]  # Sanity check for debugging: this should lead to chance perf in the test trial of 3rd episode
                        else:
                            cuepair = [params['nbcues']//2-1,params['nbcues']//2] if np.random.randint(2) else [params['nbcues']//2,params['nbcues']//2-1]
                    # else nothing, we're in the 'test' phase (which is now only 1 trial) and we sample from all pairs above

                if nb > params['bs']//2 and HALFNOBARREDPAIRUNTILT18:
                    if numtrial == 18:
                        cuepair = BARREDPAIR if np.random.randint(2) else [BARREDPAIR[1],BARREDPAIR[0]]
                    elif numtrial < 18:
                        while True:
                            cuepair =  list(np.random.choice(cuerange, 2, replace=False))
                            if (abs(cuepair[0] - cuepair[1]) == 1) :
                                if (BARREDPAIR[0] not in cuepair) or (BARREDPAIR[1] not in cuepair):
                                    break



                thistrialhascorrectorder[nb] = 1 if cuepair[0]  <  cuepair[1] else 0
                thistrialhasadjacentcues[nb] = 1 if (abs(cuepair[0]-cuepair[1]) == 1) else  0
                isadjacent_thisep[nb,numtrial]  = thistrialhasadjacentcues[nb]
                istest_thisep[nb, numtrial] = 1 if numtrial >= params['nbtraintrials'] + NBMASSEDTRIALS else 0

                # mycues = [cuepair,cuepair]
                mycues = [cuepair,]
                cuepairs_thistrial.append(cuepair)

                # Filling up other inputs for this trial
                # # We first insert some empty time steps at random either before or after the stimulus
                # for nc in range(params['triallen'] - len(mycues)  - 3):
                #     # mycues.insert(np.random.randint(len(mycues)), -1)
                #     mycues.insert(0, -1)
                # No,  we don't do that any more.

                mycues.append(params['nbcues']) # The 'go' cue, instructing response from the network
                mycues.append(-1) # One empty  step.During the first empty step, reward (computed on the previous step) is seen by the network.
                mycues.append(-1)
                # mycues.append(-1)
                assert len(mycues) == params['triallen']
                assert  mycues[NUMRESPONSESTEP] == params['nbcues']  # The 'response' step is signalled by the 'go' cue, whose number is params['nbcues'].
                cues[nb] = mycues

        cuepairs_thisep.append(cuepairs_thistrial)

        # In test period, if there ars some old cues in the store, some trials will use old cues
        if len(oldcuedata) > 0 and numtrial >= params['nbtraintrials']      + NBMASSEDTRIALS:
            for nb in range(BS):
                if np.random.rand() < PROBAOLDDATA:
                    isolddata_thisep[nb,numtrial] = 1



        # Now we are ready to actually  run  the trial:

        for numstep in range(params['triallen']):

            inputs = np.zeros((BS, params['inputsize']), dtype='float32')

            for nb in range(BS):
                # Turning the cue number for this time step into actual (signed) bitstring inputs, using the cue  data generated at the beginning of the episode - or, ocasionally, oldcuedata
                inputs[nb, :NBSTIMBITS] = 0
                if cues[nb][numstep] != -1 and cues[nb][numstep] != params['nbcues']:
                    assert len(cues[nb][numstep]) == 2
                    if isolddata_thisep[nb, numtrial]:
                        oldpos  = np.random.randint(len(oldcuedata))
                        inputs[nb, :NBSTIMBITS-1] = np.concatenate( ( oldcuedata[oldpos][nb][cues[nb][numstep][0]][:], oldcuedata[oldpos][nb][cues[nb][numstep][1]][:]  ) )
                    else:
                        inputs[nb, :NBSTIMBITS-1] = np.concatenate( ( cuedata[nb][cues[nb][numstep][0]][:], cuedata[nb][cues[nb][numstep][1]][:]  ) )

                    inputs0thistrial = inputs[0, :NBSTIMBITS-1]

                if cues[nb][numstep] == params['nbcues']:
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



            pwold =  pw.clone()

            ## Running the network
            y, v, DAout, hidden, et, pw  = net(inputsC, hidden, et, pw)  # y  should output raw scores, not probas

            hiddens0.append(hidden[0,:])

            # This should hold true if we reset h and et (not pw) between every  episode:
            # if numstep  < 2:
            #     assert torch.sum(torch.abs(pwold-pw)) < 1e-8


            if EVAL:
                allrates_thisep[:, :, numstep_ep]  = hidden.cpu().numpy()[:,:]
                ds_thisep.append(DAout.cpu().numpy())
                rs_thisep.append(reward[:, None])
                # LIMITSAVPW = 200
                if numtrial in [0,1, 18,19]:
                    allpwsavs_thisep.append(pw.cpu().numpy().astype('float16'))
                else:
                    allpwsavs_thisep.append(None)




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
                print("Tr", numtrial, "Step ", numstep, ", Cue 1  (0):", inputs[0,:params['cs']], "Cue 2 (0):", inputs[0,params['cs']:2*params['cs']],
                      "Other inputs:", inputs[0, 2*params['cs']:], "\n - Outputs(0): ", y.data.cpu().numpy()[0,:], " - action chosen(0): ", numactionschosen[0],
                        "TrialLen:", params['triallen'], "numstep", numstep, "TTHCC(0): ", thistrialhascorrectorder[0], "TTHOC(0):", isolddata_thisep[0, numtrial], "Reward (based on prev step): ", reward[0], ", DAout:", float(DAout[0]), ", cues(0):", cues[0] ) #, ", cc(0):", correctcue[0])


            # Computing the rewards. This is done for each time step.
            reward = np.zeros(BS, dtype='float32')
            for nb in range(BS):
                if numactionschosen[nb] == 1:
                    # Small penalty for any non-rest action taken
                    # In practice, this would usually be 0
                    reward[nb]  -= params['wp']

                numactionschosen_alltrialsandsteps_thisep[nb, numtrial, numstep] = numactionschosen[nb]

                if numstep == NUMRESPONSESTEP: # 2: # 4: #3: #  2:
                    # This is the 'response' step of the trial (and we showed the response signal
                    assert cues[nb][numstep] == params['nbcues']
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
            lossDA +=  torch.sum(torch.abs(DAout /  (1e-8 + net.DAmult)))   # This is a hack to "remove" DAmult from the L1 penalty. Assumes DAmult never goes  to < 0.
            lossHL1 += torch.mean(torch.abs(hidden))


            loss += (params['bent'] * y.pow(2).sum() / BS )   # In real A2c, this is an entropy incentive. Our original version of PyTorch did not have an entropy() function for Distribution, so we use sum-of-squares instead.

            numstep_ep  += 1


        # All steps done for this trial
        if numtrial >= params['nbtrials'] - params['nbtesttrials']:
            sumrewardtest += reward
            nbtesttrials += BS
            nbtesttrials_correct += np.sum(thistrialhascorrectanswer)
            nbtesttrials_adjcues += np.sum(thistrialhasadjacentcues)
            nbtesttrials_adjcues_correct += np.sum(thistrialhasadjacentcues * thistrialhascorrectanswer)
            nbtesttrials_nonadjcues += np.sum(1 - thistrialhasadjacentcues)
            nbtesttrials_nonadjcues_correct += np.sum((1-thistrialhasadjacentcues) * thistrialhascorrectanswer)


    # All trials done for this episode

    oldcuedata.append(cuedata)
    if EVAL:
        ds_thisep = np.hstack(ds_thisep)
        rs_thisep = np.hstack(rs_thisep)

    # Computing the various losses for A2C (outer-loop training)

    R = torch.zeros(BS, requires_grad=False).to(device)
    gammaR = params['gr']
    for numstepb in reversed(range(params['eplen'])) :
        R = gammaR * R + torch.from_numpy(rewards[numstepb]).detach().to(device)
        # ctrR = R - vs[numstepb][0]
        # ctrR = R - vs[numstepb]
        ctrR = R - vs[numstepb][:,0] # I think this is right...
        lossv += ctrR.pow(2).sum() / BS
        LOSSMULT  = params['testlmult'] if numstepb > params['eplen']  - params['triallen']  * params['nbtesttrials'] else 1.0

        # NOTE: We accumulate the logprobs from all time steps, even when the output is ignored (it is only used to sample response at time step 1, i.e. RESPONSETIME)
        # Unsurprisingly, performance is better if we anly record the logprobs for response time (and set them to 0 otherwise), but we keep this version because it was used in the paper.
        loss -= LOSSMULT * (logprobs[numstepb] * ctrR.detach()).sum() / BS  # Action poliy loss



    lossobj = float(loss)
    loss += params['blossv'] * lossv   # lossmult is not applied to value-prediction loss; is it right?...
    loss += params['lda'] * lossDA  # lossDA is loss on absolute value of DA output (see above)
    loss += params['lhl1']  * lossHL1
    loss /= params['eplen']
    losspw  = torch.mean(pw ** 2) * params['lpw']   # loss on squared final plastic weights is not divided by episode length
    loss += losspw

    if PRINTTRACE:
        print("lossobj (with coeff):", lossobj / params['eplen'], ", lossv (with coeff): ", params['blossv'] * float(lossv) / params['eplen'],
              "lossDA (with coeff): ", params['lda'] * float(lossDA) / params['eplen'],", losspw:", float(losspw))
        print ("Total reward for this episode(0):", sumreward[0], "Prop. of trials w/ rewarded cue:", (nbtrialswithcc / totalnbtrials),  " Total Nb of trials:", totalnbtrials)
        print("Nb Test Trials:", nbtesttrials, ", Nb Test Trials AdjCues:", nbtesttrials_adjcues, ", Nb Test Trials NonAdjCues:", nbtesttrials_nonadjcues)
        if nbtesttrials > 0:
            # Should always be the  case except for LinkedListsEval
            print("Test Perf (both methods):", np.array([nbtesttrials_correct / nbtesttrials, np.sum(iscorrect_thisep * istest_thisep) / np.sum(istest_thisep)]),
                        "Test Perf AdjCues:", np.array([(nbtesttrials_adjcues_correct / nbtesttrials_adjcues)]) if nbtesttrials_adjcues > 0 else 'N/A',
                        "Test Perf NonAdjCues:", np.array([nbtesttrials_nonadjcues_correct / nbtesttrials_nonadjcues]) if nbtesttrials_nonadjcues > 0 else 'N/A',
                        "Test perf old cues:",  np.array([np.sum(iscorrect_thisep * istest_thisep * isolddata_thisep) /  np.sum(istest_thisep * isolddata_thisep)])  if np.sum(istest_thisep * isolddata_thisep) > 0 else "N/A" ,
              )


    if not EVAL:
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), params['gc'])
        all_grad_norms.append(gn)
        if numepisode > 100:  # Burn-in period
            optimizer.step()
            if POSALPHA:
                torch.clip_(net.alpha.data, min=0)


    lossnum = float(loss)
    lossbetweensaves += lossnum
    all_losses_objective.append(lossnum)
    all_mean_rewards_ep.append(sumreward.mean())
    all_mean_testrewards_ep.append(sumrewardtest.mean())


    if PRINTTRACE:

        print("Episode", numepisode, "====")
        print("Mean loss: ", lossbetweensaves / params['pe'])
        lossbetweensaves = 0
        print("Mean reward per episode (over whole batch and last", params['pe'], "episodes: ", np.sum(all_mean_rewards_ep[-params['pe']:])/ params['pe'])
        print("Mean test-time reward per episode (over whole batch and last", params['pe'], "episodes: ", np.sum(all_mean_testrewards_ep[-params['pe']:])/ params['pe'])
        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)

        # print(" etaet: ", net.etaet.data.cpu().numpy(), " DAmult: ", net.DAmult.data.cpu().numpy(), " mean-abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())))
        print(" etaet: ", net.etaet.data.cpu().numpy(), " DAmult: ", float(net.DAmult), " mean-abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())))
        print("min/max/med-abs w, alpha, pw")
        print(float(torch.min(net.w)), float(torch.max(net.w)), float(torch.median(torch.abs(net.w))))
        print(float(torch.min(net.alpha)), float(torch.max(net.alpha)), float(torch.median(torch.abs(net.alpha))))
        print(float(torch.min(pw)), float(torch.max(pw)), float(torch.median(torch.abs(pw))))
        # pwc = pw.cpu().numpy()
        # print(np.min(pwc), np.max(pwc), np.median(np.abs(pwc)))

    # if (numepisode) % params['save_every'] == 0:
    if EVAL:
        np.savez('outcomes_'+str(numepisode)+'.npz',  c=iscorrect_thisep.astype(int), a=isadjacent_thisep.astype(int),
                 cp=np.moveaxis(np.array(cuepairs_thisep),1,0), r=resps_thisep.astype(int), ac = numactionschosen_alltrialsandsteps_thisep.astype(int))
        np.save('allrates_thisep_'+str(numepisode)+'.npy', allrates_thisep)

    if (numepisode) % params['save_every'] == 0 and numepisode  > 0:
        losslast100 = np.mean(all_losses_objective[-100:])
        print("Average loss over the last 100 episodes:", losslast100)
        print("Saving local files...")

        if (not EVAL) and numepisode > 0:
            # print("Saving model parameters...")
            # torch.save(net.state_dict(), 'net_'+suffix+'.dat')
            torch.save(net.state_dict(), 'netAE'+str(params['rngseed'])+'.dat')
            torch.save(net.state_dict(), 'net.dat')

        # with open('rewards_'+suffix+'.txt', 'w') as thefile:
        #     for item in all_mean_rewards_ep[::10]:
        #             thefile.write("%s\n" % item)
        # with open('testrew_'+suffix+'.txt', 'w') as thefile:
        #     for item in all_mean_testrewards_ep[::10]:
        #             thefile.write("%s\n" % item)
        # This is the sum of signed rewards (1 or -1) over the last nbtesttrials trials.
        # if nbtesttrials = 10, you can plot success % with this:
        # for i in range(30):
        #   plt.plot(smooth(smooth(.5 + .5*(.1*np.loadtxt('tSA'+str(1+i)+'.txt')))))
        # plt.xticks([1,1000, 2000, 3000], [1, 10000, 20000, 30000])
        with open('tAE'+str(params['rngseed'])+'.txt', 'w') as thefile:
            for item in all_mean_testrewards_ep[::10]:
                    thefile.write("%s\n" % item)


