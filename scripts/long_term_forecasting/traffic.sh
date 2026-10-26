export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=CALF

for pred_len in 96 192 336 720
do

python run.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id traffic_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0008 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss mse \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --use_prealign_moe 1 \
    --prealign_moe_num_experts 8 \
    --prealign_moe_hidden 256 \
    --prealign_moe_dropout 0.2 \
    --prealign_moe_scale 1 \
    --prealign_moe_top_k 2 \
    --prealign_moe_routing_type st_topk \
    --prealign_moe_gate_temperature 1.0 \
    --prealign_moe_lb_weight 0.01 \
    --prealign_moe_lb_type switch \
    --prealign_moe_lb_hard_weight 0.8 \
    --use_proto_route_mi_loss 1\
    --proto_route_mi_weight 0.03 \
    --proto_relation_channel_chunk_size 256 \
    --proto_route_mi_hidden 384 \
    --proto_route_mi_dropout 0.4 \
    --proto_route_mi_target soft \
    --use_proto_route_ib 0 \
    --proto_route_ib_weight 0.002 \
    --proto_route_ib_sample 1 \
    --use_expert_local_conflict 1 \
    --expert_conflict_weight 0.01 \
    --expert_conflict_start_epoch 3 \
    --expert_conflict_margin 0.10 \
    --expert_routing_margin 0.50 \
    --expert_conflict_pair_shifts 2 \
    --expert_conflict_min_grad_norm 1e-10 \
    --use_proto_route_mi_loss 0 \
    --use_gradient_density_weight 1 \
    --gradient_density_tau 1.0 \
    --gradient_density_alpha 0.5 \
    --gradient_density_max_weight 5.0 \
    --gradient_density_pair_max_weight 5.0 \
    --density_routing_margin 0.01 \
    --density_channel_chunk_size 64

echo '====================================================================================================================='
done