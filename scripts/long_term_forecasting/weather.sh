export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=CALF

for pred_len in 336 720
do
python run.py \
 --root_path ./datasets/weather/ \
 --data_path weather.csv \
 --is_training 1 \
 --task_name long_term_forecast \
 --model_id weather_${model}_${seq_len}_${pred_len} \
 --data custom \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --batch_size 64 \
 --learning_rate 0.0005 \
 --train_epochs 100 \
 --d_model 768 \
 --n_heads 4 \
 --d_ff 768 \
 --dropout 0.3 \
 --enc_in 7 \
 --c_out 7 \
 --lradj type3 \
 --gpt_layer 6 \
 --itr 1 \
 --model $model \
 --r 8 \
 --lora_alpha 32 \
 --lora_dropout 0.1 \
 --patience 5 \
 --feature_loss smooth_l1 \
 --output_loss smooth_l1 \
    --use_prealign_moe 1 \
    --prealign_moe_num_experts 6 \
    --prealign_moe_hidden 1024 \
    --prealign_moe_dropout 0.3 \
    --prealign_moe_scale 1 \
    --prealign_moe_top_k 2 \
    --prealign_moe_routing_type st_topk \
    --prealign_moe_gate_temperature 1.0 \
    --prealign_moe_lb_weight 0.003 \
    --prealign_moe_lb_type switch \
    --prealign_moe_lb_hard_weight 0.6 \
    --use_proto_route_mi_loss 1\
    --proto_route_mi_weight 0.01 \
    --proto_route_mi_start_epoch 3\
    --proto_relation_channel_chunk_size 256 \
    --proto_route_mi_hidden 512 \
    --proto_route_mi_dropout 0.55 \
    --proto_route_mi_target soft \
    --use_proto_route_ib 0 \
    --proto_route_ib_weight 0.002 \
    --proto_route_ib_sample 1 \
    --use_expert_local_conflict 1 \
    --expert_conflict_weight 0.005 \
    --expert_conflict_start_epoch 2 \
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
