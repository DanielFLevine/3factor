import argparse
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch

import wandb

from mlp import MLP, create_plastic_weights, detach_plastic_weights, clone_plastic_weights, pw_batch_size, repeat_interleave_pw
from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll, generate_batch_items_ai, generate_batch_trials_ai, generate_interleaved_ti_ai_batch, generate_grouped_ti_ai_batch
from plots import plot_pca_frozen_by_symbolic_distance, zero_shot_symbolic_distance_plot, delta_symbolic_distance_plot, ai_heatmap_plot, plot_list_linking_analysis, plot_correlation_evolution_ti, plot_correlation_evolution_ll, plot_innate_weight_analysis, plot_pair_pca_by_choice, plot_trial_violin_by_reward, plot_neural_activity_by_pair_ti, plot_neural_activity_by_pair_ll, plot_rank_neuron_analysis_ti, plot_rank_neuron_analysis_ll, plot_linking_pair_item_correlations, plot_adjacent_pair_heatmap_ti
from eval import more_items_generalization_test, mass_presentation_test, new_items_old_items_test, full_eval_ll, full_eval_ti, full_eval_ai, eval_controlled_order_ll, plastic_weight_ablation_ll, plastic_weight_ablation_ti, top_alpha_ablation, continual_learning_eval, ai_generalization_test
from losses import compute_bce_loss, compute_a2c_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output", required=False, help="Output directory for saving results")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", required=False, help="Directory for saving checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, required=False, help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5000, required=False, help="Save checkpoint every N episodes")

    # Model args
    parser.add_argument("--hidden_size", type=int, default=200, required=False, help="Size of hidden dimension")
    parser.add_argument("--extra_layers", type=int, default=0, required=False, help="Number of extra hidden layers prior to final hidden layer that combines with plastic weights")
    parser.add_argument("--plastic_weight_clip", type=float, default=None, required=False, help="Clip plastic weights to be within [-plastic_weight_clip, plastic_weight_clip]")
    parser.add_argument("--anneal_plastic_weight_clip_step", type=float, default=None, required=False, help="Amount to increase plastic_weight_clip every --anneal_plastic_weight_clip_interval episodes. Removes clip entirely once it exceeds 1e6.")
    parser.add_argument("--anneal_plastic_weight_clip_interval", type=int, default=100, required=False, help="How often (in episodes) to increase the plastic weight clip (default 100)")
    parser.add_argument("--delay_steps", type=int, default=0, required=False, help="Number of delay steps at the end of trial for an additional plastic weight update")
    parser.add_argument("--freeze_plastic_during_test", action="store_true", required=False, help="Skip Hebbian updates during test trials (plastic weights still contribute but are not updated)")
    parser.add_argument("--use_extra_neuromodulator", action="store_true", required=False, help="Use separate neuromodulator networks for each extra hidden layer")
    parser.add_argument("--simple_neuromodulator", action="store_true", required=False, help="Replace neuromodulator network with reward * single learnable weight")
    parser.add_argument("--simple_neuromodulator_bias", action="store_true", required=False, help="Add learnable bias to simple neuromodulator")
    parser.add_argument("--simple_neuromodulator_init_weight", type=float, default=1.0, required=False, help="Initial value for simple neuromodulator weight (default: 1.0)")
    parser.add_argument("--simple_neuromodulator_init_bias", type=float, default=0.0, required=False, help="Initial value for simple neuromodulator bias (default: 0.0)")
    parser.add_argument("--freeze_neuromodulator_multiplier", action="store_true", required=False, help="Freeze neuromodulator multiplier at 1.0")
    parser.add_argument("--freeze_hebbian_trace_multiplier", action="store_true", required=False, help="Freeze hebbian trace multiplier at 1.0")
    parser.add_argument("--use_answer", action="store_true", required=False, help="Use answer as reward")
    parser.add_argument("--use_sampled_choice_in_reward", action="store_true", required=False, help="Concatenate sampled choice with reward for a 2D reward embedding input")
    parser.add_argument("--greedy_sampling", action="store_true", required=False, help="Use greedy sampling (threshold at 0.5) instead of random sampling for choices")
    parser.add_argument("--use_a2c", action="store_true", required=False, help="Use A2C loss instead of BCE for outer loop training")
    parser.add_argument("--gamma", type=float, default=0.9, required=False, help="Discount factor for A2C temporal discounting")
    parser.add_argument("--value_loss_coef", type=float, default=0.1, required=False, help="Coefficient for value loss in A2C")
    parser.add_argument("--entropy_coef", type=float, default=0.1, required=False, help="Coefficient for entropy bonus in A2C")
    parser.add_argument("--use_sos_entropy", action="store_true", required=False, help="Use sum-of-squares as entropy proxy (like Miconi) instead of actual entropy")
    parser.add_argument("--plot_embeddings", action="store_true", required=False, help="Plot embeddings for eval")
    parser.add_argument("--plot_embeddings_test_only", action="store_true", required=False, help="Only plot embeddings for trial 20 (first test trial) and embedding layer at trial 0")
    parser.add_argument("--add_additional_hidden_layer_pre_plastic", action="store_true", required=False, help="Add an additional hidden layer before the first layer with plastic weights")
    parser.add_argument("--add_additional_hidden_layer_post_plastic", action="store_true", required=False, help="Add an additional hidden layer after the last layer with plastic weights")
    parser.add_argument("--scalar_alpha_layers", type=int, nargs='*', default=[], required=False, help="Layer indices that use a learnable scalar alpha instead of a matrix. 1..N=extra layers, N+1=final layer. E.g. --scalar_alpha_layers 1 2 for extra layers 1 and 2.")
    parser.add_argument("--freeze_innate_layers", type=int, nargs='*', default=[], required=False, help="Freeze innate weights for specified layers. 0=embedding, 1..N=extra hidden layers, N+1=final hidden (fc2). E.g. --freeze_innate_layers 0 1 freezes embedding and first extra layer.")
    parser.add_argument("--freeze_alpha_layers", type=int, nargs='*', default=[], required=False, help="Freeze alpha weights for specified layers. 1..N=alpha_extra[0..N-1], N+1=alpha (final layer). E.g. --freeze_alpha_layers 1 2 freezes alpha for first two extra layers.")
    parser.add_argument("--freeze_readout", action="store_true", required=False, help="Freeze readout (choice) layer weights")
    parser.add_argument("--multi_neuromodulator", type=int, default=1, required=False, help="Number of neuromodulator channels per layer (default 1)")
    parser.add_argument("--multi_neuromodulator_shared_trace", action=argparse.BooleanOptionalAction, default=True, help="Share Hebbian trace multiplier across NM channels (default True, use --no_multi_neuromodulator_shared_trace for independent)")
    parser.add_argument("--direct_readout", action="store_true", required=False, help="Merge fc2 and choice into a single layer: fc2 outputs scalar logit directly, plastic weights are (1, hidden_size)")
    parser.add_argument("--use_sigmoid", action="store_true", required=False, help="Use sigmoid instead of tanh for all nonlinearities")
    parser.add_argument("--use_capped_relu", action="store_true", required=False, help="Use ReLU capped at 1 instead of tanh for all nonlinearities")
    parser.add_argument("--single_nm_unit", action="store_true", required=False, help="Use single neuromodulator output unit instead of 2 (removes subtraction mechanism)")
    parser.add_argument("--linear_hebbian", action="store_true", required=False, help="Remove nonlinearity from Hebbian trace (use raw outer product instead of tanh)")
    parser.add_argument("--no_alpha", action="store_true", required=False, help="Remove alpha parameters entirely so plastic weights contribute directly without element-wise modulation")
    parser.add_argument("--no_embedding", action="store_true", required=False, help="Remove embedding layer so raw item vectors flow directly into the first plastic layer")
    parser.add_argument("--linear_activation", action="store_true", required=False, help="Use identity activation (no nonlinearity) in feedforward layers")
    parser.add_argument("--ones_readout", action="store_true", required=False, help="Replace readout layer with frozen all-ones weights and no bias (applies to fc2 if direct_readout, else choice layer)")
    parser.add_argument("--antisymmetric_readout", action="store_true", required=False, help="With --ones_readout, set second half of readout weights to -1 instead of +1")
    parser.add_argument("--antisymmetric_input_init", action="store_true", required=False, help="Initialize the first layer so item 2 weights are the negative of item 1 weights")
    parser.add_argument("--strong_antisymmetric_input_init", action="store_true", required=False, help="Strong antisymmetric init: [[A,-A],[-A,A]] where A is the upper-left block")
    parser.add_argument("--normal_init_std", type=float, default=None, required=False, help="Initialize all linear layer weights with normal distribution (mean=0, std=this value). Default: PyTorch default init.")
    parser.add_argument("--no_bias_layers", type=int, nargs='*', default=[], required=False, help="Remove bias from specified layers. 0=embedding layers, 1..N=extra hidden layers, N+1=fc2. E.g. --no_bias_layers 0 1 2 removes bias from embedding and first two extra layers. 0 is ignored when --no_embedding is set.")
    parser.add_argument("--no_readout_bias", action="store_true", required=False, help="Remove bias from the choice (readout) layer. Only applies when not using --direct_readout.")
    parser.add_argument("--innate_rank", type=int, default=None, required=False, help="Replace innate weight matrices (extra layers and fc2 if not ones_readout) with a sum of this many random outer products, controlling the rank.")
    parser.add_argument("--direct_nm", action="store_true", required=False, help="Use separate learnable scalars for positive and negative reward NM instead of a linear function of reward")
    parser.add_argument("--direct_nm_pos_init", type=float, default=0.0, required=False, help="Initial value for direct NM positive reward scalar (default: 0.0)")
    parser.add_argument("--direct_nm_neg_init", type=float, default=-1.0, required=False, help="Initial value for direct NM negative reward scalar (default: -1.0)")
    parser.add_argument("--freeze_direct_nm_pos", action="store_true", required=False, help="Freeze direct NM positive reward scalar")
    parser.add_argument("--freeze_direct_nm_neg", action="store_true", required=False, help="Freeze direct NM negative reward scalar")
    parser.add_argument("--plastic_embedding", action="store_true", required=False, help="Add plastic weights to embedding layer (Hebbian-updated like extra layers)")
    parser.add_argument("--disable_final_plastic", action="store_true", required=False, help="Disable plastic weights contribution and Hebbian update for the final (fc2) layer")

    # Optimizer args
    parser.add_argument("--learning_rate", type=float, default=0.0001, required=False, help="Learning rate for the optimizer/outer loop training")
    parser.add_argument("--grad_clip", type=float, default=2.0, required=False, help="Gradient clipping for the optimizer/outer loop training")
    parser.add_argument("--tbptt_steps", type=int, default=0, required=False, help="Truncated BPTT: detach plastic weights every K trials within an episode. 0 = no truncation (full BPTT).")
    parser.add_argument("--nonadj_loss_multiplier", type=float, default=1.0, required=False, help="Multiplier for non-adjacent loss in the optimizer/outer loop training")
    parser.add_argument("--mask_adjacent_loss", action="store_true", required=False, help="Only compute loss on non-adjacent pairs (mask adjacent pairs from loss)")
    
    # Task args
    parser.add_argument("--num_episodes", type=int, default=30000, required=False, help="Number of episodes to train for")
    parser.add_argument("--num_train_trials", type=int, default=64, required=False, help="Number of training trials per episode for transitive inference task")
    parser.add_argument("--num_test_trials", type=int, default=32, required=False, help="Number of test trials per episode for transitive inference task")
    parser.add_argument("--exhaustive_test_pairs", action="store_true", required=False, help="Show all N*(N-1) ordered pairs exactly once at test time (overrides --num_test_trials for TI)")
    parser.add_argument("--num_items", type=int, default=7, required=False, help="Number of items in transitive inference task")
    parser.add_argument("--item_size", type=int, default=32, required=False, help="Dimensionality of each item")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size or number of synchronous agents in each episode. Taken from A2C algorithm even though we don't use the policy loss")
    parser.add_argument("--num_episodes_per_reset", type=int, default=1, required=False, help="Number of episodes per reset of plastic weights")
    parser.add_argument("--item_range", type=int, nargs='+', default=[4, 9], required=False, help="Range of number of items in each episode")
    parser.add_argument("--arbitrary", action='store_true', required=False, help="Use both adjacent and non-adjacent pairs at test time")
    parser.add_argument("--change_items_throughout_batch", action='store_true', required=False, help="Each agent in an episode sees a different set of item representations")
    parser.add_argument("--additional_items", type=int, default=10, required=False, help="Number of additional items to test generalization on")
    parser.add_argument("--random_long_episode", action='store_true', required=False, help="Randomly inject long episodes during training")
    parser.add_argument("--full_eval_batch_size", type=int, default=2000, required=False, help="Total batch size for full evaluation with full_eval_ll")
    parser.add_argument("--put_linking_trials_first", action='store_true', required=False, help="Put linking trials first in the batch")
    parser.add_argument("--randomize_list_order", action='store_true', required=False, help="Randomly choose whether list 1 or list 2 appears first (50/50)")

    # Associative inference args
    parser.add_argument("--associative_inference_metatraining", action='store_true', required=False, help="Do associative inference metatraining")
    parser.add_argument("--associative_inference_num_groups", type=int, default=2, required=False, help="Number of groups for associative inference metatraining")
    parser.add_argument("--associative_inference_num_items_per_group", type=int, default=3, required=False, help="Number of items per group for associative inference metatraining")
    parser.add_argument("--associative_inference_ti", action='store_true', required=False, help="Mix TI and AI episodes (50/50 random per episode)")
    parser.add_argument("--ai_test_nonadj_ratio", type=float, default=-1.0, required=False, help="Ratio of nonadjacent test trials (0.0-1.0). E.g., 0.5 means 50%% nonadjacent, 50%% adjacent. -1 disables weighting.")
    parser.add_argument("--ai_num_test_trials", type=int, default=32, required=False, help="Number of test trials per episode for AI task (separate from TI's num_test_trials)")
    parser.add_argument("--ai_exclude_same_item", action='store_true', required=False, help="Exclude same-item trials [A,A] from AI training and evaluation")
    parser.add_argument("--interleave_ti_ai", action='store_true', required=False, help="Interleave TI and AI trials within the same episode (training and test phases mix both tasks)")
    parser.add_argument("--tri_mode_ti_ai", action='store_true', required=False, help="1/3 TI, 1/3 AI, 1/3 joint grouped TI+AI episodes")

    # List-linking args
    parser.add_argument("--num_trials_list_1", type=int, required=False, help="Number of trials in list 1 for list-linking task")
    parser.add_argument("--num_trials_list_2", type=int, required=False, help="Number of trials in list 2 for list-linking task")
    parser.add_argument("--num_trials_linking_pair", type=int, required=False, help="Number of trials in linking pair for list-linking task")
    parser.add_argument("--listlinking_metatraining", action='store_true', required=False, help="Do list-linking metatraining")
    
    # Other args
    parser.add_argument("--burn_in_period", type=int, default=-1, required=False, help="Number of episodes to burn in for before training")
    parser.add_argument("--full_eval_interval", type=int, default=1000, required=False, help="Interval for full evaluation with full_eval_ll")
    parser.add_argument("--ll_eval", action='store_true', required=False, help="Perform full list-linking evaluation (controlled order, ablations, etc.)")
    parser.add_argument("--ll_zero_shot", action='store_true', required=False, help="Run LL zero-shot symbolic distance plot only (without other LL evaluations)")

    # Evaluation toggle args (use --skip_* to disable specific evaluations)
    parser.add_argument("--skip_rank_neurons", action='store_true', required=False, help="Skip rank neuron analysis for TI and LL")
    parser.add_argument("--skip_neural_activity", action='store_true', required=False, help="Skip neural activity plots for TI and LL")
    parser.add_argument("--skip_correlation_evolution", action='store_true', required=False, help="Skip correlation evolution plots for TI and LL")
    parser.add_argument("--skip_plastic_weight_ablation", action='store_true', required=False, help="Skip plastic weight ablation evaluations")
    parser.add_argument("--skip_top_alpha_ablation", action='store_true', required=False, help="Skip top alpha weight ablation evaluation")
    parser.add_argument("--top_alpha_k", type=int, default=10, required=False, help="Number of top alpha weights to zero out in ablation (default: 10)")
    parser.add_argument("--skip_length_generalization", action='store_true', required=False, help="Skip length generalization test (more_items_generalization_test)")
    parser.add_argument("--skip_mass_presentation", action='store_true', required=False, help="Skip mass presentation test")
    parser.add_argument("--skip_new_items_old_items", action='store_true', required=False, help="Skip new items old items test")
    parser.add_argument("--continual_learning_eval", action='store_true', required=False, help="Run continual learning evaluation (TI<->AI, LL<->AI)")
    parser.add_argument("--continual_learning_num_networks", type=int, default=512, required=False, help="Number of networks to average over for continual learning eval (default: 512)")
    parser.add_argument("--ai_generalization_eval", action='store_true', required=False, help="Run AI generalization evaluation (test on more groups/items)")
    parser.add_argument("--ai_generalization_additional_groups", type=int, default=3, required=False, help="Number of additional groups to test beyond base (default: 3)")
    parser.add_argument("--ai_generalization_additional_items", type=int, default=3, required=False, help="Number of additional items per group to test beyond base (default: 3)")

    return parser.parse_args()

def main(args):
    # Check for conflicting flags
    if args.associative_inference_metatraining and args.associative_inference_ti:
        logger.warning("WARNING: Both associative_inference_metatraining and associative_inference_ti are set!")
        logger.warning("associative_inference_metatraining takes priority, so training will be 100% AI (not 50/50 TI+AI)")
        logger.warning("To get 50/50 TI+AI mixed training, use only --associative_inference_ti (not --associative_inference_metatraining)")
    if args.interleave_ti_ai and (args.associative_inference_metatraining or args.associative_inference_ti):
        logger.warning("WARNING: --interleave_ti_ai is set along with other AI flags!")
        logger.warning("--interleave_ti_ai takes priority and will interleave TI+AI trials within each episode")
    if args.associative_inference_ti and not args.associative_inference_metatraining and not args.interleave_ti_ai:
        logger.info("associative_inference_ti is set: will randomly mix TI and AI episodes (50/50)")
    if args.interleave_ti_ai:
        logger.info("interleave_ti_ai is set: each episode will interleave TI and AI trials in training and test phases")
    if args.tri_mode_ti_ai:
        if args.interleave_ti_ai or args.associative_inference_metatraining or args.associative_inference_ti:
            logger.warning("WARNING: --tri_mode_ti_ai is set along with other AI/TI flags!")
            logger.warning("--tri_mode_ti_ai takes priority: 1/3 TI, 1/3 AI, 1/3 joint grouped TI+AI episodes")
        logger.info("tri_mode_ti_ai is set: 1/3 TI, 1/3 AI, 1/3 joint grouped TI+AI episodes")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Override architecture args from checkpoint when resuming
    if args.resume_from_checkpoint is not None:
        ckpt_args = torch.load(args.resume_from_checkpoint, map_location=device).get('args', {})
        arch_keys = [
            'hidden_size', 'item_size', 'extra_layers', 'use_extra_neuromodulator',
            'use_answer', 'use_sampled_choice_in_reward',
            'add_additional_hidden_layer_pre_plastic', 'add_additional_hidden_layer_post_plastic',
            'scalar_alpha_layers', 'simple_neuromodulator', 'simple_neuromodulator_bias',
            'multi_neuromodulator', 'multi_neuromodulator_shared_trace',
            'direct_readout', 'use_sigmoid', 'use_capped_relu', 'single_nm_unit', 'linear_hebbian',
            'no_alpha', 'no_embedding', 'linear_activation', 'ones_readout', 'antisymmetric_readout',
            'antisymmetric_input_init', 'strong_antisymmetric_input_init', 'no_bias_layers',
            'direct_nm', 'plastic_embedding', 'disable_final_plastic', 'no_readout_bias',
        ]
        overridden = []
        for key in arch_keys:
            if key in ckpt_args and getattr(args, key, None) != ckpt_args[key]:
                overridden.append(f"{key}: {getattr(args, key, None)} -> {ckpt_args[key]}")
                setattr(args, key, ckpt_args[key])
        if overridden:
            logger.info(f"Overrode architecture args from checkpoint: {overridden}")

        # Migrate old scalar_alpha_layers convention: 0 meant final layer, now it's extra_layers+1
        if args.scalar_alpha_layers and 0 in args.scalar_alpha_layers:
            old = list(args.scalar_alpha_layers)
            args.scalar_alpha_layers = [args.extra_layers + 1 if x == 0 else x for x in args.scalar_alpha_layers]
            logger.info(f"Migrated scalar_alpha_layers from old convention: {old} -> {args.scalar_alpha_layers}")

    logger.info(f"args: {args}")
    num_episodes = args.num_episodes
    num_episodes_per_reset = args.num_episodes_per_reset
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials
    num_test_trials_final = num_test_trials
    assert num_train_trials + num_test_trials > 0
    if args.listlinking_metatraining:
        assert args.num_trials_list_1 is not None
        assert args.num_trials_list_2 is not None
        assert args.num_trials_linking_pair is not None
        num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

    item_size = args.item_size
    batch_size = args.batch_size

    input_size = 2*item_size

    wandb_name = f"mlp_hs{args.hidden_size}_lr{args.learning_rate}_extralayers{args.extra_layers}"
    if args.listlinking_metatraining:
        wandb_name += "_ll"
    if args.associative_inference_metatraining:
        wandb_name += "_ai"
    if args.associative_inference_ti:
        wandb_name += "_aiti"
    if args.interleave_ti_ai:
        wandb_name += "_interleave"
    if args.tri_mode_ti_ai:
        wandb_name += "_trimode"
    if args.use_extra_neuromodulator:
        wandb_name += "_extranms"
    if args.use_a2c:
        wandb_name += "_a2c"
    else:
        wandb_name += "_bce"
    if args.greedy_sampling:
        wandb_name += "_greedy"
    else:
        wandb_name += "_stochastic"
    if args.mask_adjacent_loss:
        wandb_name += "_maskadj"
    if args.nonadj_loss_multiplier != 1.0:
        wandb_name += f"_nonadjw{args.nonadj_loss_multiplier}"
    if args.scalar_alpha_layers:
        wandb_name += f"_scalaralpha{'_'.join(str(x) for x in args.scalar_alpha_layers)}"
    if args.simple_neuromodulator:
        wandb_name += "_simplenm"
        if args.simple_neuromodulator_bias:
            wandb_name += "bias"
        if args.simple_neuromodulator_init_weight != 1.0:
            wandb_name += f"_nmw{args.simple_neuromodulator_init_weight}"
        if args.simple_neuromodulator_init_bias != 0.0:
            wandb_name += f"_nmb{args.simple_neuromodulator_init_bias}"
    if args.freeze_neuromodulator_multiplier:
        wandb_name += "_frzNM"
    if args.freeze_hebbian_trace_multiplier:
        wandb_name += "_frzHT"
    if args.freeze_innate_layers:
        wandb_name += f"_frzInnate{'_'.join(str(x) for x in args.freeze_innate_layers)}"
    if args.freeze_alpha_layers:
        wandb_name += f"_frzAlpha{'_'.join(str(x) for x in args.freeze_alpha_layers)}"
    if args.freeze_readout:
        wandb_name += "_frzReadout"
    if args.multi_neuromodulator > 1:
        wandb_name += f"_multiNM{args.multi_neuromodulator}"
        if not args.multi_neuromodulator_shared_trace:
            wandb_name += "_indeptrace"
    if args.direct_readout:
        wandb_name += "_directreadout"
    if args.use_sigmoid:
        wandb_name += "_sigmoid"
    if args.use_capped_relu:
        wandb_name += "_cappedrelu"
    if args.single_nm_unit:
        wandb_name += "_singlenm"
    if args.linear_hebbian:
        wandb_name += "_linearhebb"
    if args.no_alpha:
        wandb_name += "_noalpha"
    if args.no_embedding:
        wandb_name += "_noEmbed"
    if args.linear_activation:
        wandb_name += "_linearact"
    if args.ones_readout:
        wandb_name += "_onesreadout"
    if args.no_readout_bias:
        wandb_name += "_noReadoutBias"
    if args.antisymmetric_readout:
        wandb_name += "_antisym"
    if args.antisymmetric_input_init:
        wandb_name += "_antisymInput"
    if args.strong_antisymmetric_input_init:
        wandb_name += "_strongAntisymInput"
    if args.normal_init_std is not None:
        wandb_name += f"_ninit{args.normal_init_std}"
    if args.no_bias_layers:
        wandb_name += f"_nobias{'_'.join(str(x) for x in args.no_bias_layers)}"
    if args.innate_rank is not None:
        wandb_name += f"_innateRank{args.innate_rank}"
    if args.direct_nm:
        wandb_name += f"_directNM_p{args.direct_nm_pos_init}_n{args.direct_nm_neg_init}"
        if args.freeze_direct_nm_pos:
            wandb_name += "_frzDNMpos"
        if args.freeze_direct_nm_neg:
            wandb_name += "_frzDNMneg"
    if args.plastic_embedding:
        wandb_name += "_plasticEmbed"
    if args.disable_final_plastic:
        wandb_name += "_noFinalPlastic"
    if args.freeze_plastic_during_test:
        wandb_name += "_frzPlasticTest"
    if args.exhaustive_test_pairs:
        wandb_name += "_exhaustiveTest"

    wandb.init(project="3factor", name=wandb_name)

    model = MLP(
        input_size=input_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        plastic_weight_clip=args.plastic_weight_clip,
        delay_steps=args.delay_steps,
        use_extra_neuromodulator=args.use_extra_neuromodulator,
        extra_layers=args.extra_layers,
        use_answer=args.use_answer,
        use_sampled_choice_in_reward=args.use_sampled_choice_in_reward,
        add_additional_hidden_layer_pre_plastic=args.add_additional_hidden_layer_pre_plastic,
        add_additional_hidden_layer_post_plastic=args.add_additional_hidden_layer_post_plastic,
        scalar_alpha_layers=args.scalar_alpha_layers,
        simple_neuromodulator=args.simple_neuromodulator,
        simple_neuromodulator_bias=args.simple_neuromodulator_bias,
        simple_neuromodulator_init_weight=args.simple_neuromodulator_init_weight,
        simple_neuromodulator_init_bias=args.simple_neuromodulator_init_bias,
        freeze_neuromodulator_multiplier=args.freeze_neuromodulator_multiplier,
        freeze_hebbian_trace_multiplier=args.freeze_hebbian_trace_multiplier,
        multi_neuromodulator=args.multi_neuromodulator,
        multi_neuromodulator_shared_trace=args.multi_neuromodulator_shared_trace,
        direct_readout=args.direct_readout,
        use_sigmoid=args.use_sigmoid,
        use_capped_relu=args.use_capped_relu,
        single_nm_unit=args.single_nm_unit,
        linear_hebbian=args.linear_hebbian,
        no_alpha=args.no_alpha,
        no_embedding=args.no_embedding,
        linear_activation=args.linear_activation,
        ones_readout=args.ones_readout,
        antisymmetric_readout=args.antisymmetric_readout,
        antisymmetric_input_init=args.antisymmetric_input_init,
        strong_antisymmetric_input_init=args.strong_antisymmetric_input_init,
        normal_init_std=args.normal_init_std,
        no_bias_layers=args.no_bias_layers,
        direct_nm=args.direct_nm,
        direct_nm_pos_init=args.direct_nm_pos_init,
        direct_nm_neg_init=args.direct_nm_neg_init,
        freeze_direct_nm_pos=args.freeze_direct_nm_pos,
        freeze_direct_nm_neg=args.freeze_direct_nm_neg,
        plastic_embedding=args.plastic_embedding,
        disable_final_plastic=args.disable_final_plastic,
        no_readout_bias=args.no_readout_bias,
    ).to(device)
    model.greedy_sampling = args.greedy_sampling

    # Replace innate weight matrices with low-rank (sum of outer products) if specified
    # Each outer product u@v^T has entry std ~1; scale to match Kaiming-like std of 1/sqrt(in_dim)
    if args.innate_rank is not None:
        r = args.innate_rank
        with torch.no_grad():
            for layer in model.extra_hidden_layers:
                out_dim, in_dim = layer.weight.shape
                scale = 1.0 / (in_dim * math.sqrt(r))
                W = torch.zeros(out_dim, in_dim)
                for _ in range(r):
                    W += torch.outer(torch.randn(out_dim), torch.randn(in_dim))
                layer.weight.copy_(W * scale)
            if not args.ones_readout:
                out_dim, in_dim = model.fc2.weight.shape
                scale = 1.0 / (in_dim * math.sqrt(r))
                W = torch.zeros(out_dim, in_dim)
                for _ in range(r):
                    W += torch.outer(torch.randn(out_dim), torch.randn(in_dim))
                model.fc2.weight.copy_(W * scale)
        logger.info(f"Initialized innate weights with rank-{r} outer products")

    # Freeze specified innate weight layers
    frozen_params = set()
    for layer_num in args.freeze_innate_layers:
        if layer_num == 0:
            if args.no_embedding:
                raise ValueError("Cannot freeze embedding layer (layer 0) when --no_embedding is set: no embedding layer exists.")
            for name, p in model.embedding_layer.named_parameters():
                p.requires_grad = False
                frozen_params.add(f"embedding_layer.{name}")
        elif 1 <= layer_num <= args.extra_layers:
            for name, p in model.extra_hidden_layers[layer_num - 1].named_parameters():
                p.requires_grad = False
                frozen_params.add(f"extra_hidden_layers.{layer_num - 1}.{name}")
        elif layer_num == args.extra_layers + 1:
            for name, p in model.fc2.named_parameters():
                p.requires_grad = False
                frozen_params.add(f"fc2.{name}")
        else:
            raise ValueError(f"Invalid innate layer index {layer_num}. Valid: 0=embedding, "
                             f"1..{args.extra_layers}=extra layers, {args.extra_layers + 1}=fc2")

    # Freeze specified alpha layers
    if not args.no_alpha:
        for layer_num in args.freeze_alpha_layers:
            if layer_num == 0:
                if args.plastic_embedding and hasattr(model, 'alpha_embed'):
                    model.alpha_embed.requires_grad = False
                    frozen_params.add("alpha_embed")
                else:
                    raise ValueError("Cannot freeze alpha for embedding (layer 0) unless --plastic_embedding is set.")
            elif 1 <= layer_num <= args.extra_layers:
                model.alpha_extra[layer_num - 1].requires_grad = False
                frozen_params.add(f"alpha_extra.{layer_num - 1}")
            elif layer_num == args.extra_layers + 1:
                model.alpha.requires_grad = False
                frozen_params.add("alpha")
            else:
                raise ValueError(f"Invalid alpha layer index {layer_num}. Valid: 0=embedding (if plastic_embedding), "
                                 f"1..{args.extra_layers}=alpha_extra, {args.extra_layers + 1}=alpha (final)")

    # Freeze readout (choice) layer weights
    if args.freeze_readout:
        if args.direct_readout:
            for name, p in model.fc2.named_parameters():
                p.requires_grad = False
                frozen_params.add(f"fc2.{name}")
        else:
            for name, p in model.choice.named_parameters():
                p.requires_grad = False
                frozen_params.add(f"choice.{name}")

    # Freeze _multi parameters (mirror freeze settings for multi-NM channels 1..N-1)
    if args.multi_neuromodulator > 1 and not args.no_alpha:
        # Freeze alpha_multi if corresponding alpha is frozen
        for layer_num in args.freeze_alpha_layers:
            if 1 <= layer_num <= args.extra_layers:
                for k in range(args.multi_neuromodulator - 1):
                    model.alpha_extra_multi[layer_num - 1][k].requires_grad = False
                    frozen_params.add(f"alpha_extra_multi.{layer_num - 1}.{k}")
            elif layer_num == args.extra_layers + 1:
                for k in range(args.multi_neuromodulator - 1):
                    model.alpha_multi[k].requires_grad = False
                    frozen_params.add(f"alpha_multi.{k}")

    if frozen_params:
        logger.info(f"Frozen parameters: {sorted(frozen_params)}")

    wandb.watch(model, log="all", log_freq=100)
    logger.info(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, eps=1e-6
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        logger.info(f"Resumed from checkpoint at episode {checkpoint['episode']} ({args.resume_from_checkpoint})")

    # Set up checkpoint save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_save_dir = os.path.join(args.checkpoint_dir, f"{wandb_name}_{timestamp}")

    # This is a bookkeeping device needed to plot symbolic distances. It's a nested list that stores trial information.
    # The outer list is indexed by batch index, and the inner list is indexed by  episode number, then finally trial number.
    # Then the trial information is stored as a dictionary with the following keys:
    # {
    #     "item_1": item_1, # first item index for the trial
    #     "item_2": item_2, # second item index for the trial
    #     "reverse_presentation": reverse_presentation, # boolean if items were presented in reverse order
    #     "model_output": model_output, # model output for the trial
    #     "correct_choice": correct_choice, # correct choice for the trial
    # }
    # Note that some fields can be deduced from other information, but we store them explicitly for convenience.
    # It's stored this way since it's easier to manipulate later on for plots.

    random_val = 1.0  # Default value for non-TI modes

    for episode in tqdm(range(start_episode, num_episodes)):
        # Anneal plastic weight clip
        if args.anneal_plastic_weight_clip_step is not None and model.plastic_weight_clip is not None:
            episodes_elapsed = episode - start_episode
            if episodes_elapsed > 0 and episodes_elapsed % args.anneal_plastic_weight_clip_interval == 0:
                model.plastic_weight_clip += args.anneal_plastic_weight_clip_step
                if model.plastic_weight_clip > 1e5:
                    model.plastic_weight_clip = None

        # Determine episode type for associative_inference_ti (random TI or AI)
        is_ai_episode = False
        is_interleaved_episode = args.interleave_ti_ai
        is_joint_episode = False
        if args.tri_mode_ti_ai:
            r = np.random.random()
            if r < 1/3:      # TI episode
                is_ai_episode = False
                num_items = np.random.randint(args.item_range[0], args.item_range[1])
            elif r < 2/3:    # AI episode
                is_ai_episode = True
                num_items = args.associative_inference_num_groups * args.associative_inference_num_items_per_group
            else:             # Joint episode
                is_joint_episode = True
                num_items = np.random.randint(args.item_range[0], args.item_range[1])
        elif args.interleave_ti_ai:
            num_items = np.random.randint(args.item_range[0], args.item_range[1])  # TI items
        elif args.associative_inference_ti:
            is_ai_episode = np.random.random() < 0.5
            if is_ai_episode:
                num_items = args.associative_inference_num_groups * args.associative_inference_num_items_per_group
            else:
                num_items = np.random.randint(args.item_range[0], args.item_range[1])
        elif args.associative_inference_metatraining:
            num_items = args.associative_inference_num_groups * args.associative_inference_num_items_per_group
        elif args.listlinking_metatraining:
            num_items = 8
        else:
            num_items = np.random.randint(args.item_range[0], args.item_range[1])
        if episode % num_episodes_per_reset == 0:
            plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, args.multi_neuromodulator, device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=model.first_plastic_input_size, plastic_embedding=args.plastic_embedding, input_size=input_size)
        else:
            plastic_weights, extra_plastic_weights, embed_pw = detach_plastic_weights(plastic_weights, extra_plastic_weights, args.multi_neuromodulator, embed_pw=embed_pw)

        if is_interleaved_episode:
            # Generate interleaved TI+AI episode
            (
                trials, correct_choices, task_labels,
                ti_pair_indices, ai_pair_indices,
                num_train_trials, num_test_trials_final,
                num_ti_train, num_ai_train
            ) = generate_interleaved_ti_ai_batch(
                num_items_ti=num_items,
                item_size=item_size,
                batch_size=batch_size,
                num_train_trials_ti=args.num_train_trials,
                num_test_trials_ti=args.num_test_trials,
                num_groups=args.associative_inference_num_groups,
                num_items_per_group=args.associative_inference_num_items_per_group,
                ai_num_test_trials=args.ai_num_test_trials,
                change_items_throughout_batch=args.change_items_throughout_batch,
                arbitrary_ti=args.arbitrary,
                ai_test_nonadj_ratio=args.ai_test_nonadj_ratio,
                ai_exclude_same_item=args.ai_exclude_same_item
            )
            task_labels = torch.tensor(task_labels, dtype=torch.long).to(device)
            # For interleaved mode, nonadjacent_trials computed per-trial based on task type
            # TI: |idx1-idx2| > 1, AI: |idx1-idx2| > 1 within group indices
            nonadjacent_trials_np = np.zeros((batch_size, num_train_trials + num_test_trials_final), dtype=bool)
            for b in range(batch_size):
                for t in range(num_train_trials + num_test_trials_final):
                    if task_labels[b, t].item() == 0:  # TI trial
                        ti_idx = ti_pair_indices[b, t]
                        if ti_idx[0] >= 0:  # Valid TI indices
                            nonadjacent_trials_np[b, t] = abs(ti_idx[0] - ti_idx[1]) > 1
                    else:  # AI trial
                        ai_idx = ai_pair_indices[b, t]
                        if ai_idx[0][0] >= 0:  # Valid AI indices
                            nonadjacent_trials_np[b, t] = abs(ai_idx[0][1] - ai_idx[1][1]) > 1
            nonadjacent_trials = torch.tensor(nonadjacent_trials_np, dtype=torch.bool).to(device)
            # Use TI pair_indices for symbolic distance bookkeeping (only TI trials have valid indices)
            pair_indices = ti_pair_indices
        elif is_joint_episode:
            # Generate grouped TI+AI episode (contiguous blocks)
            (
                trials, correct_choices, task_labels,
                ti_pair_indices, ai_pair_indices,
                num_train_trials, num_test_trials_final,
                num_ti_train, num_ai_train
            ) = generate_grouped_ti_ai_batch(
                num_items_ti=num_items,
                item_size=item_size,
                batch_size=batch_size,
                num_train_trials_ti=args.num_train_trials,
                num_test_trials_ti=args.num_test_trials,
                num_groups=args.associative_inference_num_groups,
                num_items_per_group=args.associative_inference_num_items_per_group,
                ai_num_test_trials=args.ai_num_test_trials,
                change_items_throughout_batch=args.change_items_throughout_batch,
                arbitrary_ti=args.arbitrary,
                ai_test_nonadj_ratio=args.ai_test_nonadj_ratio,
                ai_exclude_same_item=args.ai_exclude_same_item
            )
            task_labels = torch.tensor(task_labels, dtype=torch.long).to(device)
            nonadjacent_trials_np = np.zeros((batch_size, num_train_trials + num_test_trials_final), dtype=bool)
            for b in range(batch_size):
                for t in range(num_train_trials + num_test_trials_final):
                    if task_labels[b, t].item() == 0:  # TI trial
                        ti_idx = ti_pair_indices[b, t]
                        if ti_idx[0] >= 0:
                            nonadjacent_trials_np[b, t] = abs(ti_idx[0] - ti_idx[1]) > 1
                    else:  # AI trial
                        ai_idx = ai_pair_indices[b, t]
                        if ai_idx[0][0] >= 0:
                            nonadjacent_trials_np[b, t] = abs(ai_idx[0][1] - ai_idx[1][1]) > 1
            nonadjacent_trials = torch.tensor(nonadjacent_trials_np, dtype=torch.bool).to(device)
            pair_indices = ti_pair_indices
        elif args.associative_inference_metatraining or (args.associative_inference_ti and is_ai_episode) or (args.tri_mode_ti_ai and is_ai_episode):
            # Generate AI batch items and trials
            batch_items = generate_batch_items_ai(
                args.associative_inference_num_groups,
                args.associative_inference_num_items_per_group,
                item_size,
                batch_size,
                change_items_throughout_batch=args.change_items_throughout_batch
            )
            # Cap test trials at max available (all pairs excluding same item)
            total_items = args.associative_inference_num_groups * args.associative_inference_num_items_per_group
            max_test_trials = total_items * (total_items - 1)  # All ordered pairs
            ai_test_trials = args.ai_num_test_trials
            num_test_trials_ai = min(ai_test_trials, max_test_trials)
            trials, correct_choices, pair_indices, num_train_trials_ai = generate_batch_trials_ai(
                batch_items,
                args.associative_inference_num_items_per_group,
                num_test_trials_ai,
                nonadj_ratio=args.ai_test_nonadj_ratio,
                exclude_same_item=args.ai_exclude_same_item
            )
            num_train_trials = num_train_trials_ai  # Use AI-specific train trials count
            num_test_trials_final = num_test_trials_ai
            # For AI, use all trials for loss (no "nonadjacent" concept in the same way)
            # We could alternatively mask based on same-group vs different-group
            nonadjacent_trials = torch.tensor(np.abs(pair_indices[:,:,0,1] - pair_indices[:,:,1,1]) > 1, dtype=torch.bool).to(device)
        elif args.listlinking_metatraining:
            batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)
            trials, correct_choices, pair_indices = generate_batch_trials_ll(batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, num_test_trials, put_linking_trials_first=args.put_linking_trials_first, randomize_list_order=args.randomize_list_order)
            nonadjacent_trials = torch.tensor(np.abs(pair_indices[:,:,0] - pair_indices[:,:,1]) > 1, dtype=torch.bool).to(device)
        else:
            # TI episode - reset trial counts to args values (may have been overwritten by AI episode)
            num_train_trials = args.num_train_trials
            num_test_trials_final = args.num_test_trials
            batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)
            if args.exhaustive_test_pairs:
                num_test_trials_final = num_items * (num_items - 1)
            elif args.random_long_episode:
                random_val = np.random.random()
                if random_val < 0.05:
                    num_test_trials_final = args.num_test_trials * 10
            trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(batch_items, num_train_trials, num_test_trials_final, arbitrary=args.arbitrary, exhaustive_test=args.exhaustive_test_pairs)
            nonadjacent_trials = torch.tensor(np.abs(pair_indices[:,:,0] - pair_indices[:,:,1]) > 1, dtype=torch.bool).to(device)

        # Set task_labels to None for non-interleaved/non-joint modes
        if not is_interleaved_episode and not is_joint_episode:
            task_labels = None

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        correct_train_choices = 0
        correct_test_choices = 0
        # Critical pairs tracking (TI only): SD >= 2 and no end items
        correct_critical_pairs = 0
        total_critical_pairs = 0
        # Non-adjacent test tracking (AI): |idx1 - idx2| > 1
        correct_nonadj_test = 0
        total_nonadj_test = 0
        # Interleaved mode tracking: separate TI and AI accuracy
        correct_ti_train = 0
        correct_ti_test = 0
        correct_ai_train = 0
        correct_ai_test = 0
        total_ti_train = 0
        total_ti_test = 0
        total_ai_train = 0
        total_ai_test = 0

        # Collect data for loss computation
        all_choice_probs = []
        all_sampled_choices = []
        all_values = []

        for trial in range(num_train_trials + num_test_trials_final):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]

            trial_input = batch_trial

            freeze_plastic = args.freeze_plastic_during_test and trial >= num_train_trials
            output = model(trial_input, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw, freeze_plastic=freeze_plastic)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights
            embed_pw = output.embed_plastic_weights

            # Truncated BPTT: detach plastic weights every K trials
            if args.tbptt_steps > 0 and (trial + 1) % args.tbptt_steps == 0:
                plastic_weights, extra_plastic_weights, embed_pw = detach_plastic_weights(plastic_weights, extra_plastic_weights, args.multi_neuromodulator, embed_pw=embed_pw)

            if torch.isnan(output.choice).any() or (output.choice < 0).any() or (output.choice > 1).any():
                print(f"Trial {trial}: choice has invalid values - min={output.choice.min()}, max={output.choice.max()}, nan={torch.isnan(output.choice).sum()}")
                break

            choice_sampled = output.sampled_choices.squeeze(-1)
            choice_prob = output.choice.squeeze(-1)

            all_choice_probs.append(choice_prob)
            all_sampled_choices.append(choice_sampled)
            all_values.append(output.value)

            if trial < num_train_trials:
                correct_train_choices += (choice_sampled == batch_correct_choice).sum().item()
                # Track interleaved/joint TI/AI training accuracy
                if is_interleaved_episode or is_joint_episode:
                    choice_correct = (choice_sampled == batch_correct_choice).cpu().numpy()
                    trial_task_labels = task_labels[:, trial].cpu().numpy()
                    ti_mask = trial_task_labels == 0
                    ai_mask = trial_task_labels == 1
                    correct_ti_train += (choice_correct & ti_mask).sum()
                    correct_ai_train += (choice_correct & ai_mask).sum()
                    total_ti_train += ti_mask.sum()
                    total_ai_train += ai_mask.sum()
            else:
                correct_test_choices += (choice_sampled == batch_correct_choice).sum().item()
                # Track interleaved/joint TI/AI test accuracy
                if is_interleaved_episode or is_joint_episode:
                    choice_correct = (choice_sampled == batch_correct_choice).cpu().numpy()
                    trial_task_labels = task_labels[:, trial].cpu().numpy()
                    ti_mask = trial_task_labels == 0
                    ai_mask = trial_task_labels == 1
                    correct_ti_test += (choice_correct & ti_mask).sum()
                    correct_ai_test += (choice_correct & ai_mask).sum()
                    total_ti_test += ti_mask.sum()
                    total_ai_test += ai_mask.sum()
                # Track critical pairs accuracy for TI (SD >= 2, no end items)
                # For associative_inference_ti, only track when it's a TI episode
                is_ti_tracking = not args.associative_inference_metatraining and not args.listlinking_metatraining and not (args.associative_inference_ti and is_ai_episode) and not is_interleaved_episode and not is_joint_episode and not (args.tri_mode_ti_ai and is_ai_episode)
                if is_ti_tracking:
                    trial_pair_indices = pair_indices[:, trial, :]  # (batch_size, 2)
                    item1_indices = trial_pair_indices[:, 0]
                    item2_indices = trial_pair_indices[:, 1]
                    symbolic_distances = np.abs(item1_indices - item2_indices)
                    # Critical pairs: SD >= 2 and neither item is an end item (0 or num_items-1)
                    is_critical = (symbolic_distances >= 2) & (item1_indices != 0) & (item1_indices != num_items - 1) & (item2_indices != 0) & (item2_indices != num_items - 1)
                    choice_correct = (choice_sampled == batch_correct_choice).cpu().numpy()
                    correct_critical_pairs += (choice_correct & is_critical).sum()
                    total_critical_pairs += is_critical.sum()
                # Track non-adjacent test accuracy for AI (|idx1 - idx2| > 1)
                # For associative_inference_ti, only track when it's an AI episode
                is_ai_tracking = args.associative_inference_metatraining or (args.associative_inference_ti and is_ai_episode) or (args.tri_mode_ti_ai and is_ai_episode and not is_joint_episode)
                if is_ai_tracking:
                    # pair_indices for AI: (batch_size, num_trials, 2, 2) where [b, t, item, :] = [group, idx]
                    idx1 = pair_indices[:, trial, 0, 1]  # idx within group for item 1
                    idx2 = pair_indices[:, trial, 1, 1]  # idx within group for item 2
                    is_nonadj = np.abs(idx1 - idx2) > 1
                    choice_correct = (choice_sampled == batch_correct_choice).cpu().numpy()
                    correct_nonadj_test += (choice_correct & is_nonadj).sum()
                    total_nonadj_test += is_nonadj.sum()

        # Compute loss using appropriate loss function
        if args.use_a2c:
            episode_loss, loss_breakdown = compute_a2c_loss(
                choice_probs=all_choice_probs,
                sampled_choices=all_sampled_choices,
                values=all_values,
                correct_choices=correct_choices,
                nonadjacent_mask=nonadjacent_trials,
                num_train_trials=num_train_trials,
                gamma=args.gamma,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_coef,
                nonadj_loss_multiplier=args.nonadj_loss_multiplier,
                use_sos_entropy=args.use_sos_entropy,
                mask_adjacent_loss=args.mask_adjacent_loss,
            )
        else:
            episode_loss, loss_breakdown = compute_bce_loss(
                choice_probs=all_choice_probs,
                correct_choices=correct_choices,
                nonadjacent_mask=nonadjacent_trials,
                num_train_trials=num_train_trials,
                num_test_trials=num_test_trials_final,
                batch_size=batch_size,
                nonadj_loss_multiplier=args.nonadj_loss_multiplier,
                mask_adjacent_loss=args.mask_adjacent_loss,
                task_labels=task_labels,
            )

        # Stack choices for bookkeeping: shape (num_trials, batch_size) -> transpose to (batch_size, num_trials)
        all_choices_sampled = torch.stack(all_sampled_choices, dim=0).T.detach().cpu().numpy()
        # l2_plastic_weights_loss = torch.mean(plastic_weights ** 2) * 1e-1
        # episode_loss += l2_plastic_weights_loss
        # alpha_reg = torch.mean(model.alpha ** 2) * 1e-1
        # episode_loss += alpha_reg

        train_accuracy = correct_train_choices / (num_train_trials * batch_size)
        test_accuracy = correct_test_choices / (num_test_trials_final * batch_size)
        # Critical pairs test accuracy (TI only)
        critical_pairs_test_accuracy = correct_critical_pairs / total_critical_pairs if total_critical_pairs > 0 else 0.0
        # Non-adjacent test accuracy (AI only)
        nonadj_test_accuracy = correct_nonadj_test / total_nonadj_test if total_nonadj_test > 0 else 0.0
        # Interleaved accuracy
        ti_train_accuracy = correct_ti_train / total_ti_train if total_ti_train > 0 else 0.0
        ti_test_accuracy = correct_ti_test / total_ti_test if total_ti_test > 0 else 0.0
        ai_train_accuracy = correct_ai_train / total_ai_train if total_ai_train > 0 else 0.0
        ai_test_accuracy = correct_ai_test / total_ai_test if total_ai_test > 0 else 0.0

        # if episode % 100 == 0:
        #     logger.info(f"Episode {episode}, Loss: {episode_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        #     # logger.info(f"Plastic weights loss: {l2_plastic_weights_loss}, Alpha regularization: {alpha_reg}")
        #     logger.info(f"Neuromodulator values: {output.neuromodulator}, {output.neuromodulator.min().item()}, {output.neuromodulator.max().item()}, {output.neuromodulator.mean().item()}")
        #     logger.info(f"Plastic weights: {output.plastic_weights.abs().mean()}, {output.plastic_weights.abs().min().item()}, {output.plastic_weights.abs().max().item()}, {output.plastic_weights.abs().mean().item()}")
        #     logger.info(f"Choice: {choice_prob}, {choice_prob.min().item()}, {choice_prob.max().item()}, {choice_prob.mean().item()}")

        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        if episode > args.burn_in_period:
            optimizer.step()

        # Log plastic weight distributions
        if args.multi_neuromodulator > 1:
            # Multi-NM: plastic_weights is a list of tensors
            pw_data = torch.cat([pw.detach().flatten() for pw in plastic_weights]).cpu().numpy()
        else:
            pw_data = plastic_weights.detach().cpu().numpy().flatten()
        wandb_log_dict = {
            "episode_loss": episode_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "critical_pairs_test_accuracy": critical_pairs_test_accuracy,
            "nonadj_test_accuracy": nonadj_test_accuracy,
            "plastic_weights/histogram": wandb.Histogram(pw_data),
            "plastic_weights/mean_abs": float(np.mean(np.abs(pw_data))),
            "plastic_weights/std": float(np.std(pw_data)),
            "plastic_weights/min": float(np.min(pw_data)),
            "plastic_weights/max": float(np.max(pw_data)),
        }
        if model.plastic_weight_clip is not None:
            wandb_log_dict["plastic_weight_clip"] = model.plastic_weight_clip

        # Log scalar alpha values and gradients as line plots
        if args.scalar_alpha_layers and not args.no_alpha:
            if (args.extra_layers + 1) in args.scalar_alpha_layers:
                wandb_log_dict["scalar_alpha/final_layer_value"] = model.alpha.item()
                if model.alpha.grad is not None:
                    wandb_log_dict["scalar_alpha/final_layer_grad"] = model.alpha.grad.item()
            for i in range(args.extra_layers):
                if (i + 1) in args.scalar_alpha_layers:
                    wandb_log_dict[f"scalar_alpha/extra_layer{i+1}_value"] = model.alpha_extra[i].item()
                    if model.alpha_extra[i].grad is not None:
                        wandb_log_dict[f"scalar_alpha/extra_layer{i+1}_grad"] = model.alpha_extra[i].grad.item()

        # Log embedding plastic weight stats
        if args.plastic_embedding and embed_pw is not None:
            embed_pw_data = embed_pw.detach().cpu().numpy().flatten()
            wandb_log_dict.update({
                "embed_plastic_weights/mean_abs": float(np.mean(np.abs(embed_pw_data))),
                "embed_plastic_weights/std": float(np.std(embed_pw_data)),
                "embed_plastic_weights/max": float(np.max(embed_pw_data)),
            })
            if not args.no_alpha and hasattr(model, 'alpha_embed'):
                if model.alpha_embed.dim() == 0:
                    wandb_log_dict["scalar_alpha/embed_value"] = model.alpha_embed.item()
                    if model.alpha_embed.grad is not None:
                        wandb_log_dict["scalar_alpha/embed_grad"] = model.alpha_embed.grad.item()

        # Log task-specific metrics for associative_inference_ti mode
        if args.associative_inference_ti and not is_interleaved_episode:
            if is_ai_episode:
                wandb_log_dict["ai_episode_loss"] = episode_loss.item() if hasattr(episode_loss, 'item') else episode_loss
                wandb_log_dict["ai_train_accuracy"] = train_accuracy
                wandb_log_dict["ai_test_accuracy"] = test_accuracy
                wandb_log_dict["ai_nonadj_test_accuracy"] = nonadj_test_accuracy
            else:
                wandb_log_dict["ti_episode_loss"] = episode_loss.item() if hasattr(episode_loss, 'item') else episode_loss
                wandb_log_dict["ti_train_accuracy"] = train_accuracy
                wandb_log_dict["ti_test_accuracy"] = test_accuracy
                wandb_log_dict["ti_critical_pairs_accuracy"] = critical_pairs_test_accuracy

        # Log task-specific metrics for interleaved mode
        if is_interleaved_episode:
            wandb_log_dict["interleaved/ti_train_accuracy"] = ti_train_accuracy
            wandb_log_dict["interleaved/ti_test_accuracy"] = ti_test_accuracy
            wandb_log_dict["interleaved/ai_train_accuracy"] = ai_train_accuracy
            wandb_log_dict["interleaved/ai_test_accuracy"] = ai_test_accuracy

        # Log task-specific metrics for tri_mode_ti_ai
        if args.tri_mode_ti_ai:
            if is_joint_episode:
                wandb_log_dict["joint/ti_train_accuracy"] = ti_train_accuracy
                wandb_log_dict["joint/ti_test_accuracy"] = ti_test_accuracy
                wandb_log_dict["joint/ai_train_accuracy"] = ai_train_accuracy
                wandb_log_dict["joint/ai_test_accuracy"] = ai_test_accuracy
            elif is_ai_episode:
                wandb_log_dict["ai_episode_loss"] = episode_loss.item() if hasattr(episode_loss, 'item') else episode_loss
                wandb_log_dict["ai_train_accuracy"] = train_accuracy
                wandb_log_dict["ai_test_accuracy"] = test_accuracy
                wandb_log_dict["ai_nonadj_test_accuracy"] = nonadj_test_accuracy
            else:
                wandb_log_dict["ti_episode_loss"] = episode_loss.item() if hasattr(episode_loss, 'item') else episode_loss
                wandb_log_dict["ti_train_accuracy"] = train_accuracy
                wandb_log_dict["ti_test_accuracy"] = test_accuracy
                wandb_log_dict["ti_critical_pairs_accuracy"] = critical_pairs_test_accuracy

        # Log A2C-specific losses if using A2C
        if args.use_a2c:
            wandb_log_dict.update({
                "a2c/policy_loss": loss_breakdown['policy_loss'],
                "a2c/value_loss": loss_breakdown['value_loss'],
                "a2c/entropy_term": loss_breakdown['entropy_term'],
            })
        else:
            # Log BCE loss breakdown (train, test, nonadjacent)
            wandb_log_dict.update({
                "loss/train": loss_breakdown['train_loss'],
                "loss/test": loss_breakdown['test_loss'],
                "loss/nonadj": loss_breakdown['nonadj_loss'],
            })
            # Log task-specific loss breakdown for associative_inference_ti mode
            if args.associative_inference_ti and not is_interleaved_episode:
                if is_ai_episode:
                    wandb_log_dict.update({
                        "ai_loss/train": loss_breakdown['train_loss'],
                        "ai_loss/test": loss_breakdown['test_loss'],
                        "ai_loss/nonadj": loss_breakdown['nonadj_loss'],
                    })
                else:
                    wandb_log_dict.update({
                        "ti_loss/train": loss_breakdown['train_loss'],
                        "ti_loss/test": loss_breakdown['test_loss'],
                        "ti_loss/nonadj": loss_breakdown['nonadj_loss'],
                    })
            # Log task-specific loss breakdown for interleaved mode
            if is_interleaved_episode:
                wandb_log_dict.update({
                    "interleaved/ti_train_loss": loss_breakdown.get('ti_train_loss', 0.0),
                    "interleaved/ti_test_loss": loss_breakdown.get('ti_test_loss', 0.0),
                    "interleaved/ai_train_loss": loss_breakdown.get('ai_train_loss', 0.0),
                    "interleaved/ai_test_loss": loss_breakdown.get('ai_test_loss', 0.0),
                })
            # Log task-specific loss breakdown for tri_mode
            if args.tri_mode_ti_ai:
                if is_joint_episode:
                    wandb_log_dict.update({
                        "joint/ti_train_loss": loss_breakdown.get('ti_train_loss', 0.0),
                        "joint/ti_test_loss": loss_breakdown.get('ti_test_loss', 0.0),
                        "joint/ai_train_loss": loss_breakdown.get('ai_train_loss', 0.0),
                        "joint/ai_test_loss": loss_breakdown.get('ai_test_loss', 0.0),
                    })
                elif is_ai_episode:
                    wandb_log_dict.update({
                        "ai_loss/train": loss_breakdown['train_loss'],
                        "ai_loss/test": loss_breakdown['test_loss'],
                        "ai_loss/nonadj": loss_breakdown['nonadj_loss'],
                    })
                else:
                    wandb_log_dict.update({
                        "ti_loss/train": loss_breakdown['train_loss'],
                        "ti_loss/test": loss_breakdown['test_loss'],
                        "ti_loss/nonadj": loss_breakdown['nonadj_loss'],
                    })
        if extra_plastic_weights:
            if args.multi_neuromodulator > 1:
                # Multi-NM: extra_plastic_weights is list of lists
                for i in range(len(extra_plastic_weights)):
                    epw_data = torch.cat([epw.detach().flatten() for epw in extra_plastic_weights[i]]).cpu().numpy()
                    wandb_log_dict.update({
                        f"extra_plastic_weights_{i}/histogram": wandb.Histogram(epw_data),
                        f"extra_plastic_weights_{i}/mean_abs": float(np.mean(np.abs(epw_data))),
                        f"extra_plastic_weights_{i}/std": float(np.std(epw_data)),
                        f"extra_plastic_weights_{i}/min": float(np.min(epw_data)),
                        f"extra_plastic_weights_{i}/max": float(np.max(epw_data)),
                    })
            else:
                extra_pw_data = [extra_plastic_weights[i].detach().cpu().numpy().flatten() for i in range(len(extra_plastic_weights))]
                for i in range(len(extra_pw_data)):
                    wandb_log_dict.update({
                        f"extra_plastic_weights_{i}/histogram": wandb.Histogram(extra_pw_data[i]),
                        f"extra_plastic_weights_{i}/mean_abs": float(np.mean(np.abs(extra_pw_data[i]))),
                        f"extra_plastic_weights_{i}/std": float(np.std(extra_pw_data[i])),
                        f"extra_plastic_weights_{i}/min": float(np.min(extra_pw_data[i])),
                        f"extra_plastic_weights_{i}/max": float(np.max(extra_pw_data[i])),
                    })

        # Log multiplier parameters and gradients
        wandb_log_dict["multipliers/neuromodulator"] = model.neuromodulator_multiplier.item()
        wandb_log_dict["multipliers/hebbian_trace"] = model.hebbian_trace_multiplier.item()
        if model.neuromodulator_multiplier.grad is not None:
            wandb_log_dict["multipliers/neuromodulator_grad"] = model.neuromodulator_multiplier.grad.item()
        if model.hebbian_trace_multiplier.grad is not None:
            wandb_log_dict["multipliers/hebbian_trace_grad"] = model.hebbian_trace_multiplier.grad.item()

        if args.use_extra_neuromodulator:
            for i in range(len(model.neuromodulator_multiplier_extra)):
                wandb_log_dict[f"multipliers/neuromodulator_extra_{i}"] = model.neuromodulator_multiplier_extra[i].item()
                wandb_log_dict[f"multipliers/hebbian_trace_extra_{i}"] = model.hebbian_trace_multiplier_extra[i].item()
                if model.neuromodulator_multiplier_extra[i].grad is not None:
                    wandb_log_dict[f"multipliers/neuromodulator_extra_{i}_grad"] = model.neuromodulator_multiplier_extra[i].grad.item()
                if model.hebbian_trace_multiplier_extra[i].grad is not None:
                    wandb_log_dict[f"multipliers/hebbian_trace_extra_{i}_grad"] = model.hebbian_trace_multiplier_extra[i].grad.item()

        if args.simple_neuromodulator:
            wandb_log_dict["multipliers/simple_nm_weight"] = model.simple_nm_weight.item()
            if model.simple_nm_weight.grad is not None:
                wandb_log_dict["multipliers/simple_nm_weight_grad"] = model.simple_nm_weight.grad.item()
            if args.simple_neuromodulator_bias:
                wandb_log_dict["multipliers/simple_nm_bias"] = model.simple_nm_bias.item()
                if model.simple_nm_bias.grad is not None:
                    wandb_log_dict["multipliers/simple_nm_bias_grad"] = model.simple_nm_bias.grad.item()

        if args.direct_nm:
            wandb_log_dict["multipliers/direct_nm_pos"] = model.direct_nm_pos.item()
            wandb_log_dict["multipliers/direct_nm_neg"] = model.direct_nm_neg.item()
            if model.direct_nm_pos.grad is not None:
                wandb_log_dict["multipliers/direct_nm_pos_grad"] = model.direct_nm_pos.grad.item()
            if model.direct_nm_neg.grad is not None:
                wandb_log_dict["multipliers/direct_nm_neg_grad"] = model.direct_nm_neg.grad.item()

        # Full evaluation
        if episode % args.full_eval_interval == 0 and episode > 0:
            # TI-specific evaluations (skip for LL and AI)
            if not args.listlinking_metatraining and not args.associative_inference_metatraining:
                if not args.skip_length_generalization:
                    length_generalization_logging_dict, length_generalization_figs = more_items_generalization_test(args, model)
                    wandb_log_dict.update(length_generalization_logging_dict)
                    for num_items, fig in length_generalization_figs.items():
                        wandb_log_dict[f"length_generalization_{num_items}_items"] = wandb.Image(fig)

                if not args.skip_mass_presentation:
                    mass_presentation_logging_dict, mass_presentation_figs, mass_presentation_neuromodulator_fig = mass_presentation_test(args, model)
                    wandb_log_dict.update(mass_presentation_logging_dict)
                    for checkpoint, fig in mass_presentation_figs.items():
                        wandb_log_dict[f"mass_presentation_{checkpoint}"] = wandb.Image(fig)
                    if args.extra_layers > 0 and mass_presentation_neuromodulator_fig is not None:
                        wandb_log_dict["mass_presentation_neuromodulator"] = wandb.Image(mass_presentation_neuromodulator_fig)

                if not args.skip_new_items_old_items:
                    accuracies_dict, accuracies_fig, neuromodulator_fig, avg_accuracies_per_trial_fig = new_items_old_items_test(args, model)
                    wandb_log_dict.update(accuracies_dict)
                    wandb_log_dict["new_items_old_items_test_bar_plot"] = wandb.Image(accuracies_fig)
                    if args.extra_layers > 0:
                        wandb_log_dict["new_items_old_items_test_neuromodulator"] = wandb.Image(neuromodulator_fig)
                    wandb_log_dict["new_items_old_items_test_avg_accuracies_per_trial"] = wandb.Image(avg_accuracies_per_trial_fig)

            if args.associative_inference_metatraining or args.associative_inference_ti or args.interleave_ti_ai or args.tri_mode_ti_ai:
                # AI-specific evaluation (for both pure AI metatraining and mixed TI+AI)
                zero_shot_trials_ai, ai_metadata = full_eval_ai(args, model)
                # Compute overall accuracy for logging
                all_results = [r for results in zero_shot_trials_ai.values() for r in results]
                if all_results:
                    ai_zero_shot_accuracy = sum(all_results) / len(all_results)
                    wandb_log_dict["ai_zero_shot_accuracy"] = ai_zero_shot_accuracy

                # Compute nonadjacent and adjacent (training) pair accuracies
                num_items_per_group = ai_metadata['num_items_per_group']
                nonadj_results = []
                adj_results = []
                for (item1_id, item2_id), results in zero_shot_trials_ai.items():
                    idx1 = item1_id % num_items_per_group
                    idx2 = item2_id % num_items_per_group
                    if abs(idx1 - idx2) > 1:
                        nonadj_results.extend(results)
                    else:
                        adj_results.extend(results)

                if nonadj_results:
                    ai_nonadj_zero_shot_accuracy = sum(nonadj_results) / len(nonadj_results)
                    wandb_log_dict["ai_nonadj_zero_shot_accuracy"] = ai_nonadj_zero_shot_accuracy
                    logger.info(f"AI nonadjacent zero-shot accuracy: {ai_nonadj_zero_shot_accuracy:.4f} ({len(nonadj_results)} trials)")
                if adj_results:
                    ai_adj_zero_shot_accuracy = sum(adj_results) / len(adj_results)
                    wandb_log_dict["ai_adj_zero_shot_accuracy"] = ai_adj_zero_shot_accuracy
                    logger.info(f"AI adjacent (training) zero-shot accuracy: {ai_adj_zero_shot_accuracy:.4f} ({len(adj_results)} trials)")

                # AI heatmap plot
                ai_heatmap_fig = ai_heatmap_plot(zero_shot_trials_ai, ai_metadata)
                wandb_log_dict["ai_zero_shot_heatmap"] = wandb.Image(ai_heatmap_fig)

                # AI generalization test
                if args.ai_generalization_eval:
                    logger.info("Running AI generalization evaluation...")
                    ai_gen_results, ai_gen_heatmaps = ai_generalization_test(
                        args, model,
                        additional_groups=args.ai_generalization_additional_groups,
                        additional_items_per_group=args.ai_generalization_additional_items
                    )

                    # Log results and heatmaps
                    for (num_groups, num_items_per_group), metrics in ai_gen_results.items():
                        prefix = f"ai_gen/{num_groups}g_{num_items_per_group}i"
                        wandb_log_dict[f"{prefix}_overall"] = metrics['overall']
                        wandb_log_dict[f"{prefix}_adjacent"] = metrics['adjacent']
                        wandb_log_dict[f"{prefix}_nonadjacent"] = metrics['nonadjacent']

                    for (num_groups, num_items_per_group), fig in ai_gen_heatmaps.items():
                        wandb_log_dict[f"ai_gen_heatmap/{num_groups}g_{num_items_per_group}i"] = wandb.Image(fig)

            if args.associative_inference_ti or args.interleave_ti_ai or args.tri_mode_ti_ai or not args.associative_inference_metatraining:
                max_num_items = args.item_range[-1] - 1

                # LL zero-shot plot (runs if --ll_eval OR --ll_zero_shot)
                zero_shot_trials_ll = None
                if args.ll_eval or args.ll_zero_shot:
                    zero_shot_trials_ll = full_eval_ll(args, model)
                    zero_shot_fig_ll = zero_shot_symbolic_distance_plot(zero_shot_trials_ll, max_num_items)
                    wandb_log_dict["zero_shot_symbolic_distance_ll"] = wandb.Image(zero_shot_fig_ll)

                # Other LL evaluations (only if --ll_eval)
                if args.ll_eval:
                    # Controlled order experiment for chain hypothesis
                    controlled_order_results = eval_controlled_order_ll(args, model, num_networks=128, num_trials_per_pair=3)
                    wandb_log_dict["controlled_order_BG_degenerate"] = controlled_order_results['BG_degenerate_accuracy']
                    wandb_log_dict["controlled_order_BG_enriched"] = controlled_order_results['BG_enriched_accuracy']
                    wandb_log_dict["controlled_order_CF_degenerate"] = controlled_order_results['CF_degenerate_accuracy']
                    wandb_log_dict["controlled_order_CF_enriched"] = controlled_order_results['CF_enriched_accuracy']
                    logger.info(f"Controlled order - BG: degen={controlled_order_results['BG_degenerate_accuracy']:.3f}, enrich={controlled_order_results['BG_enriched_accuracy']:.3f}; CF: degen={controlled_order_results['CF_degenerate_accuracy']:.3f}, enrich={controlled_order_results['CF_enriched_accuracy']:.3f}")

                    # Plastic weight ablation for LL (only if extra_layers > 0)
                    if args.extra_layers > 0 and not args.skip_plastic_weight_ablation:
                        ablation_results_ll = plastic_weight_ablation_ll(args, model)
                        for ablation_name, ablation_trials in ablation_results_ll.items():
                            ablation_fig = zero_shot_symbolic_distance_plot(
                                ablation_trials, max_num_items,
                                title=f'LL Ablation: {ablation_name.replace("_", " ").title()}'
                            )
                            wandb_log_dict[f"ablation_ll_{ablation_name}"] = wandb.Image(ablation_fig)

                zero_shot_trials_ti = full_eval_ti(args, model)
                zero_shot_fig_ti = zero_shot_symbolic_distance_plot(zero_shot_trials_ti, max_num_items)
                wandb_log_dict["zero_shot_symbolic_distance_ti"] = wandb.Image(zero_shot_fig_ti)

                # Plastic weight ablation for TI (only if extra_layers > 0)
                if args.extra_layers > 0 and not args.skip_plastic_weight_ablation:
                    ablation_results_ti = plastic_weight_ablation_ti(args, model)
                    for ablation_name, ablation_trials in ablation_results_ti.items():
                        ablation_fig = zero_shot_symbolic_distance_plot(
                            ablation_trials, max_num_items,
                            title=f'TI Ablation: {ablation_name.replace("_", " ").title()}'
                        )
                        wandb_log_dict[f"ablation_ti_{ablation_name}"] = wandb.Image(ablation_fig)

                # Top alpha weight ablation (zeroes top k alpha weights by magnitude)
                if args.extra_layers > 0 and not args.skip_top_alpha_ablation and not args.no_alpha:
                    top_alpha_ti_trials, top_alpha_ll_trials, top_alpha_info_fig = top_alpha_ablation(args, model, k=args.top_alpha_k)
                    top_alpha_ti_fig = zero_shot_symbolic_distance_plot(
                        top_alpha_ti_trials, max_num_items,
                        title=f'TI Ablation: Top {args.top_alpha_k} Alpha Weights Zeroed'
                    )
                    wandb_log_dict["ablation_ti_top_alpha"] = wandb.Image(top_alpha_ti_fig)
                    wandb_log_dict["ablation_top_alpha_info"] = wandb.Image(top_alpha_info_fig)

                    # Delta plot for TI (baseline - ablated)
                    delta_ti_fig = delta_symbolic_distance_plot(
                        zero_shot_trials_ti, top_alpha_ti_trials, max_num_items,
                        title=f'TI Delta: Baseline - Top {args.top_alpha_k} Alpha Zeroed'
                    )
                    wandb_log_dict["ablation_ti_top_alpha_delta"] = wandb.Image(delta_ti_fig)

                    # LL ablation plots (if --ll_eval or --ll_zero_shot and we have baseline data)
                    if (args.ll_eval or args.ll_zero_shot) and zero_shot_trials_ll is not None:
                        top_alpha_ll_fig = zero_shot_symbolic_distance_plot(
                            top_alpha_ll_trials, max_num_items,
                            title=f'LL Ablation: Top {args.top_alpha_k} Alpha Weights Zeroed'
                        )
                        wandb_log_dict["ablation_ll_top_alpha"] = wandb.Image(top_alpha_ll_fig)

                        # Delta plot for LL (baseline - ablated)
                        delta_ll_fig = delta_symbolic_distance_plot(
                            zero_shot_trials_ll, top_alpha_ll_trials, max_num_items,
                            title=f'LL Delta: Baseline - Top {args.top_alpha_k} Alpha Zeroed'
                        )
                        wandb_log_dict["ablation_ll_top_alpha_delta"] = wandb.Image(delta_ll_fig)

            # Continual learning evaluation
            if args.continual_learning_eval:
                logger.info("Running continual learning evaluation...")
                continual_results, continual_ai_metadata = continual_learning_eval(
                    args, model, num_networks=args.continual_learning_num_networks
                )

                # TI -> AI: Plot TI symbolic distance and AI heatmap
                ti_then_ai_ti_fig = zero_shot_symbolic_distance_plot(
                    continual_results['ti_then_ai_ti'], args.item_range[-1] - 1,
                    title='Continual: TI→AI (TI Eval)'
                )
                wandb_log_dict["continual_ti_then_ai_ti"] = wandb.Image(ti_then_ai_ti_fig)
                ti_then_ai_ai_fig = ai_heatmap_plot(continual_results['ti_then_ai_ai'], continual_ai_metadata)
                wandb_log_dict["continual_ti_then_ai_ai"] = wandb.Image(ti_then_ai_ai_fig)

                # AI -> TI: Plot TI symbolic distance and AI heatmap
                ai_then_ti_ti_fig = zero_shot_symbolic_distance_plot(
                    continual_results['ai_then_ti_ti'], args.item_range[-1] - 1,
                    title='Continual: AI→TI (TI Eval)'
                )
                wandb_log_dict["continual_ai_then_ti_ti"] = wandb.Image(ai_then_ti_ti_fig)
                ai_then_ti_ai_fig = ai_heatmap_plot(continual_results['ai_then_ti_ai'], continual_ai_metadata)
                wandb_log_dict["continual_ai_then_ti_ai"] = wandb.Image(ai_then_ti_ai_fig)

                # LL -> AI: Plot LL symbolic distance and AI heatmap
                ll_then_ai_ll_fig = zero_shot_symbolic_distance_plot(
                    continual_results['ll_then_ai_ll'], 8,  # LL uses 8 items
                    title='Continual: LL→AI (LL Eval)'
                )
                wandb_log_dict["continual_ll_then_ai_ll"] = wandb.Image(ll_then_ai_ll_fig)
                ll_then_ai_ai_fig = ai_heatmap_plot(continual_results['ll_then_ai_ai'], continual_ai_metadata)
                wandb_log_dict["continual_ll_then_ai_ai"] = wandb.Image(ll_then_ai_ai_fig)

                # AI -> LL: Plot LL symbolic distance and AI heatmap
                ai_then_ll_ll_fig = zero_shot_symbolic_distance_plot(
                    continual_results['ai_then_ll_ll'], 8,  # LL uses 8 items
                    title='Continual: AI→LL (LL Eval)'
                )
                wandb_log_dict["continual_ai_then_ll_ll"] = wandb.Image(ai_then_ll_ll_fig)
                ai_then_ll_ai_fig = ai_heatmap_plot(continual_results['ai_then_ll_ai'], continual_ai_metadata)
                wandb_log_dict["continual_ai_then_ll_ai"] = wandb.Image(ai_then_ll_ai_fig)

                # Log overall accuracies
                for key, trials in continual_results.items():
                    all_results = [r for results in trials.values() for r in results]
                    if all_results:
                        acc = sum(all_results) / len(all_results)
                        wandb_log_dict[f"continual_{key}_accuracy"] = acc
                        logger.info(f"Continual {key} accuracy: {acc:.4f}")

            if args.plot_embeddings:
                try:
                    pca_frozen_figures = plot_pca_frozen_by_symbolic_distance(args, model)
                    for fig_name, fig in pca_frozen_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)
                except Exception as e:
                    logger.error(f"Error in plot_pca_frozen_by_symbolic_distance: {e}")

                # List linking analysis plots (if ll_eval is enabled)
                if args.ll_eval:
                    ll_figures = plot_list_linking_analysis(args, model)
                    for fig_name, fig in ll_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                    # Linking pair vs single item correlation analysis
                    ll_linking_corr_figures = plot_linking_pair_item_correlations(args, model)
                    for fig_name, fig in ll_linking_corr_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                # Correlation evolution plots for TI
                if not args.skip_correlation_evolution:
                    corr_evo_ti_figures = plot_correlation_evolution_ti(args, model)
                    for fig_name, fig in corr_evo_ti_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                    # Correlation evolution plots for LL (if ll_eval is enabled)
                    if args.ll_eval:
                        corr_evo_ll_figures = plot_correlation_evolution_ll(args, model)
                        for fig_name, fig in corr_evo_ll_figures.items():
                            wandb_log_dict[fig_name] = wandb.Image(fig)

                # Adjacent pair vs single item heatmaps for TI
                ti_adj_heatmap_figures = plot_adjacent_pair_heatmap_ti(args, model)
                for fig_name, fig in ti_adj_heatmap_figures.items():
                    wandb_log_dict[fig_name] = wandb.Image(fig)

                # Innate weight analysis for TI
                innate_ti_figures, _ = plot_innate_weight_analysis(args, model, task='ti')
                for fig_name, fig in innate_ti_figures.items():
                    wandb_log_dict[fig_name] = wandb.Image(fig)

                # Innate weight analysis for LL (if ll_eval is enabled)
                if args.ll_eval:
                    innate_ll_figures, _ = plot_innate_weight_analysis(args, model, task='ll')
                    for fig_name, fig in innate_ll_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                # Pair PCA by choice plots for TI
                pair_pca_ti_figures = plot_pair_pca_by_choice(args, model, task='ti')
                for fig_name, fig in pair_pca_ti_figures.items():
                    wandb_log_dict[fig_name] = wandb.Image(fig)

                # Pair PCA by choice plots for LL (if ll_eval is enabled)
                if args.ll_eval:
                    pair_pca_ll_figures = plot_pair_pca_by_choice(args, model, task='ll')
                    for fig_name, fig in pair_pca_ll_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                # Trial violin plots by reward for TI
                trial_violin_ti_figures = plot_trial_violin_by_reward(args, model, task='ti')
                for fig_name, fig in trial_violin_ti_figures.items():
                    wandb_log_dict[fig_name] = wandb.Image(fig)

                # Trial violin plots by reward for LL (if ll_eval is enabled)
                if args.ll_eval:
                    trial_violin_ll_figures = plot_trial_violin_by_reward(args, model, task='ll')
                    for fig_name, fig in trial_violin_ll_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                # Neural activity plots by pair for TI (4 individual networks)
                if not args.skip_neural_activity:
                    neural_activity_ti_figures = plot_neural_activity_by_pair_ti(args, model, num_networks=4)
                    for fig_name, fig in neural_activity_ti_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                    # Neural activity plots by pair for LL (if ll_eval is enabled)
                    if args.ll_eval:
                        neural_activity_ll_figures = plot_neural_activity_by_pair_ll(args, model, num_networks=4)
                        for fig_name, fig in neural_activity_ll_figures.items():
                            wandb_log_dict[fig_name] = wandb.Image(fig)

                # Rank neuron analysis for TI
                if not args.skip_rank_neurons:
                    rank_neuron_ti_figures = plot_rank_neuron_analysis_ti(args, model, num_networks=4)
                    for fig_name, fig in rank_neuron_ti_figures.items():
                        wandb_log_dict[fig_name] = wandb.Image(fig)

                    # Rank neuron analysis for LL (if ll_eval is enabled)
                    if args.ll_eval:
                        rank_neuron_ll_figures = plot_rank_neuron_analysis_ll(args, model, num_networks=4)
                        for fig_name, fig in rank_neuron_ll_figures.items():
                            wandb_log_dict[fig_name] = wandb.Image(fig)

        # Save checkpoint
        if args.save_every > 0 and (episode + 1) % args.save_every == 0:
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        wandb.log(wandb_log_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
