#!/bin/bash
#SBATCH --job-name=plot_ckpt
#SBATCH --output=logs/plot_ckpt_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=daniel.levine@yale.edu
#SBATCH --partition=bigmem
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb
#SBATCH --time=4:00:00

module load miniconda
module load CUDA/12.8
conda activate 3factor
cd /gpfs/radev/home/dfl32/project/3factor/mlp

export WANDB_DIR="/gpfs/radev/pi/dijk/dfl32/3factor/wandb"
export CP_2HIDDEN="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_maskadj_20260215_125858/checkpoint_ep149999.pt"
export CP_1HIDDEN="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers0_bce_greedy_maskadj_20260213_131541/checkpoint_ep149999.pt"
export CP_2HIDDEN_SCALARALPHA2_FZSCALARALPHA="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_maskadj_scalaralpha0_simplenmbias_frzNM_frzHT_frzInnate0_1_2_frzAlpha1_20260222_014534/checkpoint_ep199999.pt"
export CP_2HIDDEN_SCALARALPHA2_FZHETEROALPHA="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_maskadj_scalaralpha0_simplenmbias_frzNM_frzHT_frzInnate0_1_2_frzAlpha2_20260222_024027/checkpoint_ep199999.pt"
export CP_2HIDDEN_SCALARALPHA12="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_singlenm_20260228_021537/checkpoint_ep134999.pt"
export CP_2HIDDEN_MINIMAL="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_extranms_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_nobias1_2_20260303_184052/checkpoint_ep319999.pt"
export CP_2HIDDEN_ULTRAMINIMAL="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers1_bce_greedy_scalaralpha0_1_simplenmbias_nmw2.0_nmb-1.0_frzNM_frzHT_frzInnate1_2_frzAlpha1_2_frzReadout_directreadout_linearhebb_noEmbed_linearact_onesreadout_strongAntisymInput_ninit0.1_nobias1_2_20260304_144605/checkpoint_ep14999.pt"
export CP_0LAYER_EMBBIAS_ONESREADOUT="/gpfs/radev/pi/dijk/dfl32/3factor/checkpoints/mlp_hs200_lr0.0001_extralayers0_bce_greedy_scalaralpha1_frzNM_frzHT_frzAlpha1_directreadout_linearhebb_linearact_onesreadout_nobias1_directNM_p0.0_n-1.0_20260306_143327/checkpoint_ep1469999.pt"



python plot_from_checkpoint.py \
    --checkpoint $CP_0LAYER_EMBBIAS_ONESREADOUT \
    --wandb_run_name corr_evo_ep1469999_0layer_embbias_onesreadout \
    --num_train_trials 20 \
    --num_aggregate_episodes 10000 \
    --ll_num_trials_list_1 10 \
    --ll_num_trials_list_2 10 \
    --ll_num_trials_linking_pair 4 \
    --skip_plots \
        ti_adj_logit_evo \
        ti_adj_logit_delta \
        ti_agg_delta_mean \
        ti_agg_delta_sem \
        ti_agg_nm_by_trial \
        ti_agg_delta_by_type_mean \
        ti_agg_delta_by_type_sem \
        ti_agg_post_train_dot_mean \
        ti_agg_post_train_dot_sem \
        ti_agg_post_train_corr_mean \
        ti_agg_post_train_corr_sem \
        ti_agg_post_train_nonadj_corr_mean \
        ti_agg_post_train_cross_dot_mean \
        ti_agg_post_train_cross_dot_sem \
        ti_agg_post_train_cross_corr_mean \
        ti_agg_post_train_cross_corr_sem \
        ti_agg_dot \
        ti_agg_pair_dot \
        ti_agg_pair_pair \
        ti_agg_pair_logit \
        ti_agg_post_train_pairwise \
        ti_agg_post_train_logits \
        ti_corr_evo \
        ti_item_logit \
        ti_logit_vs_nm \
        ti_pair_embedding \
        ti_nm_by_trial \
        ti_agg_pair_pc1 \
        pca_frozen \
        ti_agg_symbolic_distance_nonlinear \
        alpha_weights_item1_vs_item2 \
        ti_agg_symbolic_distance_single_neuron_bias \
        innate_weights_top_vs_bottom \
        ll_agg_symbolic_distance \
        innate_weights_item1_vs_item2 \
        weight_heatmaps \
        weight_histograms \
        weight_heatmaps_2sigma \
        ti_agg_symbolic
