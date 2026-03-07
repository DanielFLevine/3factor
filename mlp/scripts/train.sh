#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=daniel.levine@yale.edu
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=2-00:00:00

module load miniconda
module load CUDA/12.8
conda activate 3factor
cd /gpfs/radev/home/dfl32/project/3factor/mlp

export WANDB_DIR="/gpfs/radev/pi/dijk/dfl32/3factor/wandb"

export CP_MULTINM2="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_aiti_bce_greedy_multiNM2_20260225_041155/checkpoint_ep199999.pt"
export CP_MINIMAL_LL="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_20260304_144643/checkpoint_ep499999.pt"
export CP_INNATE_RANK5="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_innateRank5_20260305_135155/checkpoint_ep999999.pt"
export CP_INNATE_RANK4="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_innateRank4_20260305_145906/checkpoint_ep999999.pt"
export CP_INNATE_RANK3="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_innateRank3_20260305_032936/checkpoint_ep499999.pt"
export CP_INNATE_RANK2="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_innateRank2_20260305_032936/checkpoint_ep499999.pt"
export CP_0LAYER_EMB_READOUT="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers0_bce_greedy_scalaralpha0_frzNM_frzHT_frzAlpha1_directreadout_linearhebb_linearact_nobias1_directNM_p0.0_n-1.0_20260305_151307/checkpoint_ep154999.pt"
export CP_0LAYER_EMBBIAS_ONESREADOUT="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers0_bce_greedy_scalaralpha0_frzNM_frzHT_frzAlpha1_frzReadout_directreadout_linearhebb_linearact_onesreadout_nobias1_directNM_p0.0_n-1.0_20260305_172001/checkpoint_ep999999.pt"


python main.py \
    --output_dir /gpfs/radev/pi/dijk/dfl32/3factor/outputs \
    --checkpoint_dir /gpfs/radev/pi/dijk/dfl32/3factor/checkpoints \
    --save_every 5000 \
    --hidden_size 30 \
    --extra_layers 0 \
    --full_eval_interval 1000 \
    --num_episodes 2000000 \
    --num_train_trials 20 \
    --num_test_trials 10 \
    --ai_num_test_trials 10 \
    --ai_test_nonadj_ratio -1.0 \
    --item_size 15 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --grad_clip 2.0 \
    --num_episodes_per_reset 1 \
    --item_range 8 9 \
    --nonadj_loss_multiplier 1.0 \
    --arbitrary \
    --additional_items 10 \
    --num_trials_list_1 10 \
    --num_trials_list_2 10 \
    --num_trials_linking_pair 4 \
    --greedy_sampling \
    --associative_inference_num_groups 2 \
    --associative_inference_num_items_per_group 3 \
    --skip_rank_neurons \
    --skip_neural_activity \
    --skip_correlation_evolution \
    --skip_plastic_weight_ablation \
    --skip_new_items_old_items \
    --skip_top_alpha_ablation \
    --ll_zero_shot \
    --skip_length_generalization \
    --skip_mass_presentation \
    --scalar_alpha_layers 1 \
    --freeze_alpha_layers 1 \
    --freeze_neuromodulator_multiplier \
    --freeze_hebbian_trace_multiplier \
    --linear_hebbian \
    --no_bias_layers 1 \
    --direct_readout \
    --linear_activation \
    --ones_readout \
    --direct_nm \
    --direct_nm_pos_init 0.0 \
    --direct_nm_neg_init -1.0 
    
    

