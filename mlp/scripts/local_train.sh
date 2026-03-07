module load miniconda
module load CUDA/12.8
conda activate 3factor
cd /gpfs/radev/home/dfl32/project/3factor/mlp

python main.py \
    --output_dir /gpfs/radev/pi/dijk/dfl32/3factor/outputs \
    --hidden_size 200 \
    --extra_layers 1 \
    --delay_steps 0 \
    --full_eval_interval 500 \
    --num_episodes 30000 \
    --num_train_trials 20 \
    --num_test_trials 10 \
    --item_size 15 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --grad_clip 2.0 \
    --num_episodes_per_reset 1 \
    --item_range 4 9 \
    --burn_in_period 100 \
    --nonadj_loss_multiplier 1.0 \
    --mask_adjacent_loss \
    --arbitrary \
    --additional_items 10 \
    --num_trials_list_1 10 \
    --num_trials_list_2 10 \
    --num_trials_linking_pair 4 \
    --greedy_sampling \
    --gamma 0.9 \
    --value_loss_coef 0.1 \
    --entropy_coef 0.1 \
    --use_sos_entropy \
    --ll_eval \
    --associative_inference_num_groups 2 \
    --associative_inference_num_items_per_group 3 \
    --ai_test_nonadj_ratio 0.5 \
    --plot_embeddings \
    --plot_embeddings_test_only
