
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from peft import LoraConfig, TaskType, get_peft_model

# from models.GPT2_arch import AccustumGPT2Model


# class ResidualExpert(nn.Module):
#     def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float = 0.0):
#         super().__init__()
#         self.down = nn.Linear(hidden_dim, bottleneck_dim)
#         self.act = nn.GELU()
#         self.up = nn.Linear(bottleneck_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: [B, M, D]
#         return self.up(self.dropout(self.act(self.down(x))))


# class ChannelHardMoE(nn.Module):
#     """
#     Channel-level hard Top-K MoE. In this patch it is applied before the channel self-attention encoder.

#     Input:
#         z: [B, M, D], where M is the number of channels / variables.
#     Routing:
#         Each channel token z[b, m] independently selects Top-K experts.
#         The gate shape is [B, M, K], not [B, K].
#     Output:
#         z_out: [B, M, D]
#         aux_loss: load-balance loss
#     """
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_experts: int = 4,
#         bottleneck_dim: int = 128,
#         dropout: float = 0.0,
#         scale: float = 0.2,
#         top_k: int = 1,
#         routing_type: str = "st_topk",
#         gate_temperature: float = 1.0,
#         lb_type: str = "switch",
#         lb_soft_weight: float = 1.0,
#         lb_hard_weight: float = 1.0,
#         lb_eps: float = 1e-8,
#     ):
#         super().__init__()
#         self.num_experts = int(num_experts)
#         self.scale = float(scale)
#         self.top_k = max(1, int(top_k))
#         self.routing_type = str(routing_type).lower()
#         self.gate_temperature = max(float(gate_temperature), 1e-6)
#         self.lb_type = str(lb_type).lower()
#         self.lb_soft_weight = float(lb_soft_weight)
#         self.lb_hard_weight = float(lb_hard_weight)
#         self.lb_eps = float(lb_eps)

#         valid_routing = {"hard_topk", "st_topk"}
#         if self.routing_type not in valid_routing:
#             raise ValueError(f"Unknown routing_type={self.routing_type}. Valid choices: {sorted(valid_routing)}")

#         valid_lb_types = {"importance", "switch", "combined"}
#         if self.lb_type not in valid_lb_types:
#             raise ValueError(f"Unknown lb_type={self.lb_type}. Valid choices: {sorted(valid_lb_types)}")

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, self.num_experts)
#         )
#         self.experts = nn.ModuleList(
#             [ResidualExpert(hidden_dim, bottleneck_dim, dropout=dropout) for _ in range(self.num_experts)]
#         )

#         # Detached copies for logging / analysis.
#         self.last_gate_soft_weights = None      # [B, M, K]
#         self.last_routing_weights = None        # [B, M, K]
#         self.last_lb_loss = None
#         self.last_lb_importance_loss = None
#         self.last_lb_switch_loss = None

#     def _build_topk_routing(self, gate_probs):
#         """
#         gate_probs: [B, M, K]
#         return routing_weights: [B, M, K]

#         hard_topk:
#             forward uses sparse Top-K normalized probabilities;
#             no task-loss gradient flows through the Top-K mask.

#         st_topk:
#             forward is still hard/sparse Top-K;
#             backward uses straight-through gradients from the dense soft gate.
#         """
#         K = gate_probs.size(-1)
#         top_k = min(self.top_k, K)

#         vals, idx = torch.topk(gate_probs, k=top_k, dim=-1)        # [B, M, top_k]
#         hard = torch.zeros_like(gate_probs).scatter_(-1, idx, vals)
#         hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-12)

#         if self.routing_type == "st_topk":
#             return hard.detach() - gate_probs.detach() + gate_probs
#         return hard.detach()

#     def _load_balance_loss(self, gate_probs, routing_weights):
#         """
#         Channel-level load balance.

#         gate_probs:      differentiable soft gate probabilities, [B, M, K]
#         routing_weights: sparse Top-K routing used by experts, [B, M, K]

#         importance:
#             K * sum_k mean_{B,M}(p_k)^2

#         switch:
#             K * sum_k stopgrad(f_k) * mean_{B,M}(p_k),
#             where f_k is the actual channel-token dispatch fraction.
#         """
#         K = gate_probs.size(-1)
#         if K <= 1:
#             zero = gate_probs.new_zeros(())
#             return zero, zero, zero

#         eps = self.lb_eps

#         prob_mean = gate_probs.float().mean(dim=(0, 1))  # [K]
#         prob_mean = prob_mean / prob_mean.sum().clamp_min(eps)
#         importance_loss = K * torch.sum(prob_mean ** 2)

#         dispatch_mask = (routing_weights.detach().float() > 0).float()  # [B, M, K]
#         dispatch_frac = dispatch_mask.mean(dim=(0, 1))                  # [K]
#         dispatch_frac = dispatch_frac / dispatch_frac.sum().clamp_min(eps)
#         switch_loss = K * torch.sum(dispatch_frac.detach() * prob_mean)

#         if self.lb_type == "importance":
#             lb_loss = self.lb_soft_weight * importance_loss
#         elif self.lb_type == "switch":
#             lb_loss = self.lb_hard_weight * switch_loss
#         else:
#             lb_loss = self.lb_soft_weight * importance_loss + self.lb_hard_weight * switch_loss

#         return lb_loss, importance_loss, switch_loss

#     def forward(self, z):
#         # z: [B, M, D]
#         z_norm = self.norm(z)
#         gate_logits = self.gate(z_norm) / self.gate_temperature       # [B, M, K]
#         gate_probs = torch.softmax(gate_logits, dim=-1)               # [B, M, K]
#         routing_weights = self._build_topk_routing(gate_probs)        # [B, M, K]

#         # Streaming weighted sum; avoid constructing [B, M, K, D].
#         mixed = torch.zeros_like(z)
#         for k, expert in enumerate(self.experts):
#             expert_out = expert(z)                                    # [B, M, D]
#             weight = routing_weights[..., k].unsqueeze(-1)            # [B, M, 1]
#             mixed = mixed + weight * expert_out

#         z_out = z + self.scale * mixed

#         lb_loss, importance_loss, switch_loss = self._load_balance_loss(gate_probs, routing_weights)
#         aux_loss = lb_loss

#         self.last_gate_soft_weights = gate_probs.detach()
#         self.last_routing_weights = routing_weights.detach()
#         self.last_lb_loss = lb_loss.detach()
#         self.last_lb_importance_loss = importance_loss.detach()
#         self.last_lb_switch_loss = switch_loss.detach()

#         return z_out, aux_loss, lb_loss


# class Encoder_PCA(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         word_embedding,
#         hidden_dim=768,
#         num_heads=12,
#         num_encoder_layers=1,
#         use_prealign_moe=False,
#         prealign_moe_num_experts=4,
#         prealign_moe_hidden=128,
#         prealign_moe_dropout=0.0,
#         prealign_moe_scale=0.2,
#         prealign_moe_top_k=1,
#         prealign_moe_routing_type="st_topk",
#         prealign_moe_gate_temperature=1.0,
#         prealign_moe_lb_type="switch",
#         prealign_moe_lb_soft_weight=1.0,
#         prealign_moe_lb_hard_weight=1.0,
#         prealign_moe_lb_eps=1e-8,
#         use_proto_route_mi_loss=False,
#         proto_route_mi_weight=1.0,
#         proto_relation_channel_chunk_size=16,
#         proto_route_mi_hidden=128,
#         proto_route_mi_dropout=0.0,
#         proto_route_mi_target="soft",
#         use_proto_route_ib=False,
#         proto_route_ib_weight=1e-3,
#         proto_route_ib_sample=True,
#     ):
#         super(Encoder_PCA, self).__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
#         self.word_embedding = word_embedding.T

#         self.current_epoch = 0
#         self.total_train_epochs = 1

#         # Only the losses used by the current method are kept:
#         #   1) channel-level MoE load-balance loss
#         #   2) route-MI distillation loss
#         #   3) optional variational information bottleneck KL loss
#         self.last_moe_lb_loss = None
#         self.last_proto_route_mi_loss = None
#         self.last_proto_route_mi_raw_loss = None
#         self.last_proto_route_mi_weight = None
#         self.last_proto_route_ce_loss = None
#         self.last_proto_route_ib_loss = None

#         self.use_proto_route_mi_loss = bool(use_proto_route_mi_loss)
#         self.proto_route_mi_weight = float(proto_route_mi_weight)
#         self.proto_route_mi_eps = 1e-8
#         self.proto_relation_channel_chunk_size = max(1, int(proto_relation_channel_chunk_size))
#         self.proto_route_mi_hidden = int(proto_route_mi_hidden)
#         self.proto_route_mi_dropout = float(proto_route_mi_dropout)
#         self.proto_route_mi_target = str(proto_route_mi_target).lower()
#         if self.proto_route_mi_target not in {"soft", "hard"}:
#             raise ValueError("proto_route_mi_target must be either 'soft' or 'hard'")

#         self.use_proto_route_ib = bool(use_proto_route_ib)
#         self.proto_route_ib_weight = float(proto_route_ib_weight)
#         self.proto_route_ib_sample = bool(proto_route_ib_sample)

#         self.use_prealign_moe = bool(use_prealign_moe)
#         if self.use_prealign_moe:
#             self.prealign_moe = ChannelHardMoE(
#                 hidden_dim=hidden_dim,
#                 num_experts=prealign_moe_num_experts,
#                 bottleneck_dim=prealign_moe_hidden,
#                 dropout=prealign_moe_dropout,
#                 scale=prealign_moe_scale,
#                 top_k=prealign_moe_top_k,
#                 routing_type=prealign_moe_routing_type,
#                 gate_temperature=prealign_moe_gate_temperature,
#                 lb_type=prealign_moe_lb_type,
#                 lb_soft_weight=prealign_moe_lb_soft_weight,
#                 lb_hard_weight=prealign_moe_lb_hard_weight,
#                 lb_eps=prealign_moe_lb_eps,
#             )
#         else:
#             self.prealign_moe = None

#         # q_phi(z|v). z has the same dimension as v, so route-MI and IB share
#         # exactly the same route decoder p_psi(g|.).
#         #   without IB: g_hat = decoder(v)
#         #   with IB:    z ~ q_phi(z|v), g_hat = decoder(z)
#         if self.use_proto_route_ib:
#             self.proto_route_ib_encoder = nn.Linear(hidden_dim, 2 * hidden_dim)
#         else:
#             self.proto_route_ib_encoder = None

#         self.proto_route_decoder = nn.Sequential(
#             nn.Linear(hidden_dim, int(proto_route_mi_hidden)),
#             nn.GELU(),
#             nn.Dropout(float(proto_route_mi_dropout)),
#             nn.Linear(int(proto_route_mi_hidden), int(prealign_moe_num_experts)),
#         )

#     def set_current_epoch(self, epoch: int, total_epochs: int = None):
#         self.current_epoch = int(epoch)
#         if total_epochs is not None:
#             self.total_train_epochs = max(1, int(total_epochs))

#     def _proto_route_mi_schedule_weight(self):
#         """Enable route-MI/IB only in the second half of training."""
#         total = max(1, int(getattr(self, 'total_train_epochs', 1)))
#         start_epoch = total // 2 + 1
#         return 1.0 if int(self.current_epoch) >= start_epoch else 0.0

#     def _encode_route_bottleneck(self, route_repr):
#         """Encode route-distillation representation into bottleneck latent z.

#         route_repr: [B, C, D]
#             In the current version this is the cross-attention output that is
#             directly sent to the LLM.
#         Returns:
#             z:       [B, C, D], same dimension as route_repr so the route decoder is shared.
#             kl_loss: scalar, mean KL(q_phi(z|route_repr) || N(0,I))
#         """
#         if not self.use_proto_route_ib or self.proto_route_ib_encoder is None:
#             return route_repr, route_repr.new_zeros(())

#         stats = self.proto_route_ib_encoder(route_repr)
#         mu, logvar = torch.chunk(stats, chunks=2, dim=-1)
#         logvar = torch.clamp(logvar, min=-10.0, max=10.0)

#         if self.training and self.proto_route_ib_sample:
#             eps = torch.randn_like(mu)
#             z = mu + torch.exp(0.5 * logvar) * eps
#         else:
#             z = mu

#         kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
#         kl_loss = kl.sum(dim=-1).mean() / max(1, int(mu.size(-1)))
#         return z, kl_loss

#     def _route_mi_loss(self, aligned_repr):
#         """Route-MI distillation on the representation that is directly sent to the LLM.

#         aligned_repr: [B, M, D]
#             Output of semantic prototype cross-attention. This is the same
#             representation that will be returned as the text/LLM-side token
#             representation, rather than the indirect prototype-weighted vector
#             a_{b,m}E.

#         MoE routing target:
#             g_{b,m} = prealign_moe.last_gate_soft_weights, if target='soft'
#             g_{b,m} = prealign_moe.last_routing_weights,   if target='hard'

#         Optional information bottleneck:
#             without IB: g_hat = decoder(aligned_repr)
#             with IB:    z ~ q_phi(z|aligned_repr), g_hat = decoder(z)

#         Loss:
#             L = CE(sg(g), p_psi(g|aligned_repr or z))
#                 + beta_IB * KL(q_phi(z|aligned_repr)||N(0,I))
#         """
#         if aligned_repr is None or self.prealign_moe is None:
#             return self.word_embedding.new_zeros(())

#         if self.proto_route_mi_target == "hard":
#             route_target = getattr(self.prealign_moe, "last_routing_weights", None)
#         else:
#             route_target = getattr(self.prealign_moe, "last_gate_soft_weights", None)
#         if route_target is None:
#             return aligned_repr.new_zeros(())

#         eps = self.proto_route_mi_eps
#         B, M, D = aligned_repr.shape
#         route_target = route_target.detach().to(device=aligned_repr.device, dtype=aligned_repr.dtype)
#         route_target = route_target.clamp_min(eps)
#         route_target = route_target / route_target.sum(dim=-1, keepdim=True).clamp_min(eps)

#         if route_target.shape[0] != B or route_target.shape[1] != M:
#             raise RuntimeError(
#                 f"Route-MI shape mismatch: aligned_repr has shape {tuple(aligned_repr.shape)}, "
#                 f"but route_target has shape {tuple(route_target.shape)}. "
#                 "Route-MI expects both tensors to be channel-level [B, M, *]."
#             )

#         chunk_size = max(1, min(int(self.proto_relation_channel_chunk_size), M))
#         total_ce_loss = aligned_repr.new_zeros(())
#         total_ib_loss = aligned_repr.new_zeros(())
#         total_count = 0
#         total_chunks = 0

#         for start in range(0, M, chunk_size):
#             end = min(M, start + chunk_size)

#             # Directly use the cross-attention output token that will be sent
#             # to the LLM, instead of reconstructing a semantic vector by
#             # sem_vec = attn_weights @ proto_embed.
#             route_repr = aligned_repr[:, start:end, :].float()
#             route_student, ib_loss = self._encode_route_bottleneck(route_repr)

#             logits = self.proto_route_decoder(route_student)
#             log_prob = F.log_softmax(logits, dim=-1)

#             target = route_target[:, start:end, :].float()
#             ce = -torch.sum(target * log_prob, dim=-1)
#             total_ce_loss = total_ce_loss + ce.sum()
#             total_ib_loss = total_ib_loss + ib_loss
#             total_count += ce.numel()
#             total_chunks += 1

#         ce_loss = total_ce_loss / max(total_count, 1)
#         ib_loss = total_ib_loss / max(total_chunks, 1)
#         total_loss = ce_loss + self.proto_route_ib_weight * ib_loss

#         self.last_proto_route_ce_loss = ce_loss.detach()
#         self.last_proto_route_ib_loss = ib_loss.detach()
#         return total_loss

#     def forward(self, x):
#         B = x.shape[0]

#         word_embedding = self.word_embedding
#         if word_embedding.ndim == 2:
#             word_embedding = word_embedding.unsqueeze(0).expand(B, -1, -1)
#         elif word_embedding.shape[0] != B:
#             word_embedding = word_embedding[0:1].expand(B, -1, -1)

#         x = self.linear(x)  # [B, M, D]

#         moe_aux_loss = x.new_zeros(())
#         moe_lb_loss = x.new_zeros(())
#         proto_route_mi_loss = x.new_zeros(())
#         proto_route_mi_raw_loss = x.new_zeros(())
#         proto_route_mi_weight = x.new_zeros(())

#         self.last_moe_lb_loss = moe_lb_loss
#         self.last_proto_route_mi_loss = proto_route_mi_loss
#         self.last_proto_route_mi_raw_loss = proto_route_mi_raw_loss
#         self.last_proto_route_mi_weight = proto_route_mi_weight
#         self.last_proto_route_ce_loss = x.new_zeros(())
#         self.last_proto_route_ib_loss = x.new_zeros(())

#         if self.use_prealign_moe:
#             x, moe_aux_loss, moe_lb_loss = self.prealign_moe(x)
#             self.last_moe_lb_loss = moe_lb_loss.detach()

#         x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
#         x_time = x

#         q = x.transpose(0, 1)
#         k = v = word_embedding.transpose(0, 1)
#         x, attn_weights = self.cross_attention(
#             q, k, v,
#             # Route-MI now uses the cross-attention output x directly, so
#             # attention weights are no longer needed for the auxiliary loss.
#             need_weights=False,
#             average_attn_weights=True,
#         )

#         if self.use_proto_route_mi_loss:
#             schedule_weight = float(self._proto_route_mi_schedule_weight())
#             proto_route_mi_weight = torch.tensor(schedule_weight, device=x.device, dtype=x.dtype)
#             if schedule_weight > 0.0:
#                 aligned_repr = x.transpose(0, 1)  # [B, M, D], directly sent to the LLM after this layer
#                 proto_route_mi_raw_loss = self._route_mi_loss(aligned_repr)
#                 proto_route_mi_loss = proto_route_mi_raw_loss * proto_route_mi_weight
#                 moe_aux_loss = moe_aux_loss + self.proto_route_mi_weight * proto_route_mi_loss

#             self.last_proto_route_mi_raw_loss = proto_route_mi_raw_loss.detach()
#             self.last_proto_route_mi_weight = proto_route_mi_weight.detach()
#             self.last_proto_route_mi_loss = proto_route_mi_loss.detach()

#         x = x.transpose(0, 1)
#         return x_time, x, moe_aux_loss, moe_lb_loss, proto_route_mi_loss


# class Model(nn.Module):
#     def __init__(self, configs, device):
#         super(Model, self).__init__()
#         self.pred_len = configs.pred_len
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=configs.r,
#             lora_alpha=configs.lora_alpha,
#             lora_dropout=configs.lora_dropout,
#             target_modules=["c_attn"]
#         )
#         self.task_name = configs.task_name
#         self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
#         self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)

#         self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
#         self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
#         self.gpt2 = get_peft_model(self.gpt2, peft_config)

#         word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)

#         for _, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name or 'wpe' in name or 'lora' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#         for _, (name, param) in enumerate(self.gpt2_text.named_parameters()):
#             if 'wpe' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#         self.time_proj = nn.ModuleList(
#             [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
#         )
#         self.text_proj = nn.ModuleList(
#             [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
#         )

#         self.in_layer = Encoder_PCA(
#             configs.seq_len,
#             word_embedding,
#             hidden_dim=configs.d_model,
#             use_prealign_moe=getattr(configs, 'use_prealign_moe', 0) == 1,
#             prealign_moe_num_experts=getattr(configs, 'prealign_moe_num_experts', 4),
#             prealign_moe_hidden=getattr(configs, 'prealign_moe_hidden', 128),
#             prealign_moe_dropout=getattr(configs, 'prealign_moe_dropout', 0.0),
#             prealign_moe_scale=getattr(configs, 'prealign_moe_scale', 0.2),
#             prealign_moe_top_k=getattr(configs, 'prealign_moe_top_k', 1),
#             prealign_moe_routing_type=getattr(configs, 'prealign_moe_routing_type', 'st_topk'),
#             prealign_moe_gate_temperature=getattr(configs, 'prealign_moe_gate_temperature', 1.0),
#             prealign_moe_lb_type=getattr(configs, 'prealign_moe_lb_type', 'switch'),
#             prealign_moe_lb_soft_weight=getattr(configs, 'prealign_moe_lb_soft_weight', 1.0),
#             prealign_moe_lb_hard_weight=getattr(configs, 'prealign_moe_lb_hard_weight', 1.0),
#             prealign_moe_lb_eps=getattr(configs, 'prealign_moe_lb_eps', 1e-8),
#             use_proto_route_mi_loss=getattr(configs, 'use_proto_route_mi_loss', 0) == 1,
#             proto_route_mi_weight=getattr(configs, 'proto_route_mi_weight', 1.0),
#             proto_relation_channel_chunk_size=getattr(configs, 'proto_relation_channel_chunk_size', 16),
#             proto_route_mi_hidden=getattr(configs, 'proto_route_mi_hidden', 128),
#             proto_route_mi_dropout=getattr(configs, 'proto_route_mi_dropout', 0.0),
#             proto_route_mi_target=getattr(configs, 'proto_route_mi_target', 'soft'),
#             use_proto_route_ib=getattr(configs, 'use_proto_route_ib', 0) == 1,
#             proto_route_ib_weight=getattr(configs, 'proto_route_ib_weight', 1e-3),
#             proto_route_ib_sample=getattr(configs, 'proto_route_ib_sample', 1) == 1,
#         )

#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
#         elif self.task_name == 'classification':
#             self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
#         elif self.task_name == 'imputation':
#             self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
#         elif self.task_name == 'anomaly_detection':
#             self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

#         for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
#             layer.to(device=device)
#             layer.train()

#         self.cnt = 0

#     def set_current_epoch(self, epoch: int, total_epochs: int = None):
#         if hasattr(self.in_layer, 'set_current_epoch'):
#             self.in_layer.set_current_epoch(epoch, total_epochs=total_epochs)

#     def forecast(self, x):
#         B, L, M = x.shape
#         means = x.mean(1, keepdim=True).detach()
#         x = x - means
#         stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
#         x /= stdev

#         x = rearrange(x, 'b l m -> b m l')
#         outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

#         outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
#         outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

#         outputs_time += outputs_time1
#         outputs_text += outputs_text1

#         intermidiate_feat_time = tuple(
#             [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
#         )
#         intermidiate_feat_text = tuple(
#             [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
#         )

#         outputs_time = self.out_layer(outputs_time[:, -M:, :])
#         outputs_text = self.out_layer(outputs_text[:, -M:, :])

#         outputs_time = rearrange(outputs_time, 'b m l -> b l m')
#         outputs_text = rearrange(outputs_text, 'b m l -> b l m')

#         outputs_text = outputs_text * stdev + means
#         outputs_time = outputs_time * stdev + means

#         return {
#             'outputs_text': outputs_text,
#             'outputs_time': outputs_time,
#             'intermidiate_time': intermidiate_feat_time,
#             'intermidiate_text': intermidiate_feat_text,
#             'moe_aux_loss': moe_aux_loss,
#             'moe_lb_loss': moe_lb_loss,
#             'proto_route_mi_loss': proto_route_mi_loss,
#             'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
#             'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
#             'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
#             'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
#         }

#     def classification(self, x):
#         B, L, M = x.shape
#         x = rearrange(x, 'b l m -> b m l')
#         outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

#         outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
#         outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

#         outputs_time += outputs_time1
#         outputs_text += outputs_text1

#         intermidiate_feat_time = tuple(
#             [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
#         )
#         intermidiate_feat_text = tuple(
#             [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
#         )

#         outputs_time = outputs_time.reshape(B, -1)
#         outputs_text = outputs_text.reshape(B, -1)
#         outputs_time = self.out_layer(outputs_time)
#         outputs_text = self.out_layer(outputs_text)

#         return {
#             'outputs_text': outputs_text,
#             'outputs_time': outputs_time,
#             'intermidiate_time': intermidiate_feat_time,
#             'intermidiate_text': intermidiate_feat_text,
#             'moe_aux_loss': moe_aux_loss,
#             'moe_lb_loss': moe_lb_loss,
#             'proto_route_mi_loss': proto_route_mi_loss,
#             'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
#             'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
#             'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
#             'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
#         }

#     def imputation(self, x, mask):
#         B, L, M = x.shape
#         means = x.mean(1, keepdim=True).detach()
#         x = x - means
#         x = x.masked_fill(mask == 0, 0)
#         stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
#         x /= stdev

#         x = rearrange(x, 'b l m -> b m l')
#         outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

#         outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
#         outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

#         outputs_time += outputs_time1
#         outputs_text += outputs_text1

#         intermidiate_feat_time = tuple(
#             [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
#         )
#         intermidiate_feat_text = tuple(
#             [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
#         )

#         outputs_time = self.out_layer(outputs_time)
#         outputs_text = self.out_layer(outputs_text)

#         outputs_time = rearrange(outputs_time, 'b m l -> b l m')
#         outputs_text = rearrange(outputs_text, 'b m l -> b l m')

#         outputs_text = outputs_text * stdev + means
#         outputs_time = outputs_time * stdev + means

#         return {
#             'outputs_text': outputs_text,
#             'outputs_time': outputs_time,
#             'intermidiate_time': intermidiate_feat_time,
#             'intermidiate_text': intermidiate_feat_text,
#             'moe_aux_loss': moe_aux_loss,
#             'moe_lb_loss': moe_lb_loss,
#             'proto_route_mi_loss': proto_route_mi_loss,
#             'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
#             'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
#             'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
#             'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
#         }

#     def anomaly_detection(self, x):
#         B, L, M = x.shape
#         means = x.mean(1, keepdim=True).detach()
#         x = x - means
#         stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
#         x /= stdev

#         x = rearrange(x, 'b l m -> b m l')
#         outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

#         outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
#         outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

#         outputs_time += outputs_time1
#         outputs_text += outputs_text1

#         intermidiate_feat_time = tuple(
#             [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
#         )
#         intermidiate_feat_text = tuple(
#             [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
#         )

#         outputs_time = self.out_layer(outputs_time)
#         outputs_text = self.out_layer(outputs_text)

#         outputs_time = rearrange(outputs_time, 'b m l -> b l m')
#         outputs_text = rearrange(outputs_text, 'b m l -> b l m')

#         outputs_text = outputs_text * stdev + means
#         outputs_time = outputs_time * stdev + means

#         return {
#             'outputs_text': outputs_text,
#             'outputs_time': outputs_time,
#             'intermidiate_time': intermidiate_feat_time,
#             'intermidiate_text': intermidiate_feat_text,
#             'moe_aux_loss': moe_aux_loss,
#             'moe_lb_loss': moe_lb_loss,
#             'proto_route_mi_loss': proto_route_mi_loss,
#             'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
#             'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
#             'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
#             'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
#         }

#     def forward(self, x, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             output = self.forecast(x)
#         elif self.task_name == 'classification':
#             output = self.classification(x)
#         elif self.task_name == "imputation":
#             output = self.imputation(x, mask)
#         elif self.task_name == "anomaly_detection":
#             output = self.anomaly_detection(x)
#         else:
#             output = self.forecast(x)
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model

from models.GPT2_arch import AccustumGPT2Model


class ResidualExpert(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float = 0.0, scale: float = 0.2):
        super().__init__()
        self.scale = float(scale)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Cached tensors for Expert-local conflict (ELC). They are temporary
        # graph references, overwritten by every forward pass.
        self.last_input = None
        self.last_preact = None
        self.last_hidden = None
        self.last_output = None

    def forward(self, x):
        # x: [N_k, D] for sparse dispatch, or [B, M, D].
        self.last_input = x
        preact = self.down(x)
        self.last_preact = preact
        hidden = self.dropout(self.act(preact))
        self.last_hidden = hidden
        expert_output = self.up(hidden)
        # Cache the unscaled MLP output so ELC derivatives include self.scale
        # for both the up and down projections.
        self.last_output = expert_output
        return x + self.scale * expert_output


class ChannelHardMoE(nn.Module):
    """
    Channel-level hard Top-K MoE. In this patch it is applied before the channel self-attention encoder.

    Input:
        z: [B, M, D], where M is the number of channels / variables.
    Routing:
        Each channel token z[b, m] independently selects Top-K experts.
        The gate shape is [B, M, K], not [B, K].
    Output:
        z_out: [B, M, D]
        aux_loss: load-balance loss
    """
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        bottleneck_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 0.2,
        top_k: int = 1,
        routing_type: str = "st_topk",
        gate_temperature: float = 1.0,
        lb_type: str = "switch",
        lb_soft_weight: float = 1.0,
        lb_hard_weight: float = 1.0,
        lb_eps: float = 1e-8,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.scale = float(scale)
        self.top_k = max(1, int(top_k))
        self.routing_type = str(routing_type).lower()
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.lb_type = str(lb_type).lower()
        self.lb_soft_weight = float(lb_soft_weight)
        self.lb_hard_weight = float(lb_hard_weight)
        self.lb_eps = float(lb_eps)

        valid_routing = {"hard_topk", "st_topk"}
        if self.routing_type not in valid_routing:
            raise ValueError(f"Unknown routing_type={self.routing_type}. Valid choices: {sorted(valid_routing)}")

        valid_lb_types = {"importance", "switch", "combined"}
        if self.lb_type not in valid_lb_types:
            raise ValueError(f"Unknown lb_type={self.lb_type}. Valid choices: {sorted(valid_lb_types)}")

        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_experts)
        )
        self.experts = nn.ModuleList(
            [ResidualExpert(hidden_dim, bottleneck_dim, dropout=dropout, scale=self.scale)
             for _ in range(self.num_experts)]
        )

        # Detached copies for logging / analysis.
        self.last_gate_soft_weights = None      # [B, M, K]
        self.last_routing_weights = None        # [B, M, K]
        self.last_lb_loss = None
        self.last_lb_importance_loss = None
        self.last_lb_switch_loss = None

        # State used by ELC. Actual computation still follows sparse Top-K;
        # the conflict regularizer uses a dense soft gate recomputed from a
        # detached copy of the router input.
        self.last_router_input = None
        # Shared input projection W_in = Encoder_PCA.linear. Both tensors are
        # cached for gradient-density estimation and overwritten on each pass.
        # The projection output is a *non-detached* graph tensor used in VJP.
        self.last_density_projection_input = None
        self.last_density_projection_output = None
        self.last_dispatch_mask = None
        self.last_selected_flat_indices = None

    def _build_topk_routing(self, gate_probs):
        """
        gate_probs: [B, M, K]
        return routing_weights: [B, M, K]

        hard_topk:
            forward uses sparse Top-K normalized probabilities;
            no task-loss gradient flows through the Top-K mask.

        st_topk:
            forward is still hard/sparse Top-K;
            backward uses straight-through gradients from the dense soft gate.
        """
        K = gate_probs.size(-1)
        top_k = min(self.top_k, K)

        vals, idx = torch.topk(gate_probs, k=top_k, dim=-1)        # [B, M, top_k]
        hard = torch.zeros_like(gate_probs).scatter_(-1, idx, vals)
        hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if self.routing_type == "st_topk":
            return hard.detach() - gate_probs.detach() + gate_probs
        return hard.detach()

    def _load_balance_loss(self, gate_probs, routing_weights):
        """
        Channel-level load balance.

        gate_probs:      differentiable soft gate probabilities, [B, M, K]
        routing_weights: sparse Top-K routing used by experts, [B, M, K]

        importance:
            K * sum_k mean_{B,M}(p_k)^2

        switch:
            K * sum_k stopgrad(f_k) * mean_{B,M}(p_k),
            where f_k is the actual channel-token dispatch fraction.
        """
        K = gate_probs.size(-1)
        if K <= 1:
            zero = gate_probs.new_zeros(())
            return zero, zero, zero

        eps = self.lb_eps

        prob_mean = gate_probs.float().mean(dim=(0, 1))  # [K]
        prob_mean = prob_mean / prob_mean.sum().clamp_min(eps)
        importance_loss = K * torch.sum(prob_mean ** 2)

        dispatch_mask = (routing_weights.detach().float() > 0).float()  # [B, M, K]
        dispatch_frac = dispatch_mask.mean(dim=(0, 1))                  # [K]
        dispatch_frac = dispatch_frac / dispatch_frac.sum().clamp_min(eps)
        switch_loss = K * torch.sum(dispatch_frac.detach() * prob_mean)

        if self.lb_type == "importance":
            lb_loss = self.lb_soft_weight * importance_loss
        elif self.lb_type == "switch":
            lb_loss = self.lb_hard_weight * switch_loss
        else:
            lb_loss = self.lb_soft_weight * importance_loss + self.lb_hard_weight * switch_loss

        return lb_loss, importance_loss, switch_loss

    def _soft_gate_for_regularizer(self):
        """Dense soft routing distribution used only by ELC.

        The router input is detached, so the ELC term updates the router-side
        LayerNorm/gate MLP but does not directly alter upstream representations.
        """
        if self.last_router_input is None:
            return None
        z_norm = self.norm(self.last_router_input)
        gate_logits = self.gate(z_norm) / self.gate_temperature
        return torch.softmax(gate_logits, dim=-1)

    @staticmethod
    @torch.no_grad()
    def _gradient_density_weights(projection_delta, projection_input,
                                  bandwidth=1.0, alpha=0.5,
                                  max_weight=5.0, eps=1e-8,
                                  channel_chunk_size=64):
        """Gaussian KDE over implicit per-channel gradients of W_in.

        W_in: [D,L]; G[b,c] = dL/dz[b,c] outer input[b,c] (shape [D,L]).
        We avoid materializing [B,C,D,L]. For two normalized rank-1
        gradients, their inner product is the product of the normalized
        delta and input inner products.

        Note: dL/dz[b,c] comes from the scalar forecasting probe. Because
        downstream attention mixes channels, this is the channel-specific
        contribution to the global forecasting gradient, not the exact
        gradient of an isolated output-channel loss ell[b,c].

        Kernel includes the diagonal self-term, as in the supplied formula.
        Density and weights are fully detached teacher statistics.
        """
        if bandwidth <= 0:
            raise ValueError('gradient density bandwidth must be positive')
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('gradient density alpha must be in [0,1]')
        if max_weight < 1.0:
            raise ValueError('gradient density max_weight must be >= 1')
        if projection_delta.shape[:2] != projection_input.shape[:2]:
            raise ValueError('projection gradient/input batch-channel sizes differ')
        if projection_delta.ndim != 3 or projection_input.ndim != 3:
            raise ValueError('density inputs must have shape [B,C,D] / [B,C,L]')
        # Full float32 arithmetic even under mixed precision.
        delta = projection_delta.detach().float()
        inp = projection_input.detach().float()
        B, C = inp.shape[:2]
        if B == 0 or C == 0:
            raise ValueError('cannot estimate gradient density for empty batch/channels')
        densities = []
        # Mathematically exact chunking over independent channels (no sampling).
        step = max(1, int(channel_chunk_size))
        for start in range(0, C, step):
            d = delta[:, start:start+step, :].transpose(0, 1).contiguous()  # [Ck,B,D]
            x = inp[:, start:start+step, :].transpose(0, 1).contiguous()   # [Ck,B,L]
            dn = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
            xn = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            d_unit = d / dn.clamp_min(eps)
            x_unit = x / xn.clamp_min(eps)
            # ||G/(||G||+eps)||, including zero-gradient handling.
            gn = dn * xn
            scale = gn / (gn + eps)
            grad_dot = torch.bmm(d_unit, d_unit.transpose(1, 2))
            input_dot = torch.bmm(x_unit, x_unit.transpose(1, 2))
            gram = grad_dot * input_dot
            unit_inner = gram * (scale * scale.transpose(1, 2))
            squared_norm = scale.square()
            dist_sq = (squared_norm + squared_norm.transpose(1, 2)
                       - 2.0 * unit_inner).clamp_min(0.0)
            kernel = torch.exp(-dist_sq / (2.0 * bandwidth * bandwidth))
            densities.append(kernel.mean(dim=-1).transpose(0, 1))
        density = torch.cat(densities, dim=1)  # [B,C]
        if alpha == 0.0:
            weights = torch.ones_like(density)
        else:
            raw = (density + eps).pow(-alpha).clamp(max=max_weight)
            # Clamp denominator (rather than add eps) so alpha=0 is exactly 1.
            weights = raw / raw.mean(dim=0, keepdim=True).clamp_min(eps)
        stats = {
            'density_mean': float(density.mean().item()),
            'density_min': float(density.min().item()),
            'density_max': float(density.max().item()),
            'sample_weight_min': float(weights.min().item()),
            'sample_weight_max': float(weights.max().item()),
            'sample_weight_mean': float(weights.mean().item()),
        }
        return weights.detach(), stats

    @staticmethod
    def _full_mlp_grad_cosine(
        delta_out_a, delta_out_b,
        hidden_a, hidden_b,
        delta_pre_a, delta_pre_b,
        input_a, input_b,
        eps=1e-12,
    ):
        """Exact cosine for concatenated [W_down,b_down,W_up,b_up] gradients.

        For a Linear layer, a per-token weight gradient is delta x^T. Its
        Frobenius inner product can therefore be obtained from delta/input
        inner products without materializing a per-token weight-gradient matrix.
        """
        do_dot = (delta_out_a * delta_out_b).sum(dim=-1)
        h_dot = (hidden_a * hidden_b).sum(dim=-1)
        dp_dot = (delta_pre_a * delta_pre_b).sum(dim=-1)
        x_dot = (input_a * input_b).sum(dim=-1)
        inner = do_dot * h_dot + do_dot + dp_dot * x_dot + dp_dot

        do_a2 = delta_out_a.pow(2).sum(dim=-1)
        do_b2 = delta_out_b.pow(2).sum(dim=-1)
        h_a2 = hidden_a.pow(2).sum(dim=-1)
        h_b2 = hidden_b.pow(2).sum(dim=-1)
        dp_a2 = delta_pre_a.pow(2).sum(dim=-1)
        dp_b2 = delta_pre_b.pow(2).sum(dim=-1)
        x_a2 = input_a.pow(2).sum(dim=-1)
        x_b2 = input_b.pow(2).sum(dim=-1)

        norm_a2 = do_a2 * (h_a2 + 1.0) + dp_a2 * (x_a2 + 1.0)
        norm_b2 = do_b2 * (h_b2 + 1.0) + dp_b2 * (x_b2 + 1.0)
        denom = torch.sqrt(norm_a2.clamp_min(0.0) * norm_b2.clamp_min(0.0)).clamp_min(eps)
        return (inner / denom).clamp(-1.0, 1.0), torch.sqrt(norm_a2.clamp_min(0.0)), torch.sqrt(norm_b2.clamp_min(0.0))

    def expert_local_conflict_loss(
        self,
        probe_loss,
        conflict_margin=0.1,
        routing_margin=0.5,
        pair_shifts=1,
        min_grad_norm=1e-10,
        use_gradient_density=False,
        density_bandwidth=1.0,
        density_alpha=0.5,
        density_max_weight=5.0,
        density_pair_max_weight=5.0,
        density_routing_margin=0.01,
        density_channel_chunk_size=64,
    ):
        """Compute Expert-local conflict loss for the current mini-batch.

        * Top-K dispatch from the real forward pass defines which channel
          tokens currently share an expert.
        * Only same-channel pairs across different batch windows are compared.
        * Conflict margin: cosine < -conflict_margin is treated as harmful.
        * Routing margin: once dense soft-routing cosine <= routing_margin,
          the conflicting pair receives no further repulsive pressure.
        * Gradient conflict is detached and acts only as a teacher signal;
          therefore this auxiliary loss introduces no second-order gradients.
        """
        soft_gate = self._soft_gate_for_regularizer()
        if soft_gate is None or self.last_dispatch_mask is None:
            z = probe_loss * 0.0
            return z, {'valid_pairs': 0.0, 'conflict_pairs': 0.0,
                       'conflict_ratio': 0.0, 'route_similarity': 0.0,
                       'gradient_cosine': 0.0}

        if self.last_selected_flat_indices is None:
            z = soft_gate.sum() * 0.0
            return z, {'valid_pairs': 0.0, 'conflict_pairs': 0.0,
                       'conflict_ratio': 0.0, 'route_similarity': 0.0,
                       'gradient_cosine': 0.0}

        targets = []
        active_experts = []
        for m, expert in enumerate(self.experts):
            idx = self.last_selected_flat_indices[m]
            if idx is None or idx.numel() == 0 or expert.last_output is None or expert.last_preact is None:
                continue
            active_experts.append(m)
            targets.extend([expert.last_output, expert.last_preact])

        if not targets:
            z = soft_gate.sum() * 0.0
            return z, {'valid_pairs': 0.0, 'conflict_pairs': 0.0,
                       'conflict_ratio': 0.0, 'route_similarity': 0.0,
                       'gradient_cosine': 0.0}

        # One autograd traversal obtains both expert-local derivatives and
        # the gradient of the shared input projection (only in DWGC mode).
        if use_gradient_density:
            if self.last_density_projection_output is None or self.last_density_projection_input is None:
                raise RuntimeError('DWGC needs W_in projection input/output cache from Encoder_PCA')
            if density_pair_max_weight <= 0:
                raise ValueError('density_pair_max_weight must be positive')
            targets.append(self.last_density_projection_output)
        grads = torch.autograd.grad(
            probe_loss, targets, retain_graph=True, create_graph=False,
            allow_unused=True,
        )

        density_stats = {}
        sample_density_weights = None
        if use_gradient_density:
            projection_delta = grads[-1]
            if projection_delta is None:
                raise RuntimeError('DWGC projection output has no forecasting gradient')
            sample_density_weights, density_stats = self._gradient_density_weights(
                projection_delta,
                self.last_density_projection_input,
                bandwidth=float(density_bandwidth),
                alpha=float(density_alpha),
                max_weight=float(density_max_weight),
                channel_chunk_size=int(density_channel_chunk_size),
            )

        numerator = soft_gate.sum() * 0.0
        conflict_weight_sum = soft_gate.new_zeros(())
        valid_pair_count = soft_gate.new_zeros(())
        conflict_pair_count = soft_gate.new_zeros(())
        conflict_route_sum = soft_gate.new_zeros(())
        valid_grad_cos_sum = soft_gate.new_zeros(())
        density_expert_overlap_sum = soft_gate.new_zeros(())
        density_active_pair_count = soft_gate.new_zeros(())

        dispatch = self.last_dispatch_mask
        B = soft_gate.size(0)
        max_shift = min(max(1, int(pair_shifts)), max(B - 1, 1))

        grad_offset = 0
        full_input = self.last_router_input.detach()
        flat_size = full_input.size(0) * full_input.size(1)
        input_dim = full_input.size(-1)

        for m in active_experts:
            expert = self.experts[m]
            delta_out_sel = grads[grad_offset]
            delta_pre_sel = grads[grad_offset + 1]
            grad_offset += 2
            if delta_out_sel is None or delta_pre_sel is None:
                continue

            idx = self.last_selected_flat_indices[m]
            # Scatter selected-token tensors back to [B,M,*] only for conflict
            # analysis. Unselected locations remain zero and are ignored by the
            # dispatch mask, so this does not change forward computation.
            delta_out_flat = delta_out_sel.detach().new_zeros((flat_size, delta_out_sel.size(-1)))
            delta_out_flat.index_copy_(0, idx, delta_out_sel.detach())
            delta_pre_flat = delta_pre_sel.detach().new_zeros((flat_size, delta_pre_sel.size(-1)))
            delta_pre_flat.index_copy_(0, idx, delta_pre_sel.detach())
            hidden_sel = expert.last_hidden.detach()
            hidden_flat = hidden_sel.new_zeros((flat_size, hidden_sel.size(-1)))
            hidden_flat.index_copy_(0, idx, hidden_sel)

            delta_out = delta_out_flat.view(B, -1, delta_out_sel.size(-1))
            delta_pre = delta_pre_flat.view(B, -1, delta_pre_sel.size(-1))
            hidden = hidden_flat.view(B, -1, hidden_sel.size(-1))
            inp = full_input.view(B, -1, input_dim)

            for shift in range(1, max_shift + 1):
                if B <= shift:
                    break
                same_expert = dispatch[:-shift, :, m] & dispatch[shift:, :, m]
                if not torch.any(same_expert):
                    continue

                grad_cos, norm_a, norm_b = self._full_mlp_grad_cosine(
                    delta_out[:-shift], delta_out[shift:],
                    hidden[:-shift], hidden[shift:],
                    delta_pre[:-shift], delta_pre[shift:],
                    inp[:-shift], inp[shift:],
                )
                valid = same_expert & (norm_a > float(min_grad_norm)) & (norm_b > float(min_grad_norm))
                if not torch.any(valid):
                    continue

                conflict = F.relu(-grad_cos - float(conflict_margin)).detach()
                conflict_mask = valid & (conflict > 0)

                # IMPORTANT: ELC uses the full dense soft routing distribution;
                # Top-K is used only for actual expert execution/dispatch.
                route_sim = F.cosine_similarity(
                    soft_gate[:-shift], soft_gate[shift:], dim=-1, eps=1e-8
                )
                if use_gradient_density:
                    # Formula in the supplied figures:
                    # w_ij = min(w_pair_max, sqrt(w_i * w_j)), with stopgrad.
                    pair_w = (sample_density_weights[:-shift]
                              * sample_density_weights[shift:]).clamp_min(0.0).sqrt()
                    pair_w = pair_w.clamp(max=float(density_pair_max_weight))
                    pair_w = pair_w * valid.to(pair_w.dtype)
                    # Per-expert co-assignment probability, NOT a cosine over
                    # the full expert distribution (which v2 ELC uses).
                    expert_overlap = soft_gate[:-shift, :, m] * soft_gate[shift:, :, m]
                    route_penalty = F.relu(expert_overlap - float(density_routing_margin))
                    numerator = numerator + (pair_w * conflict * route_penalty).sum()
                    density_expert_overlap_sum = density_expert_overlap_sum + (
                        expert_overlap * conflict_mask.to(expert_overlap.dtype)).sum()
                    density_active_pair_count = density_active_pair_count + (
                        conflict_mask & (route_penalty > 0)).sum()
                    # Figure formula normalizes by sum of pair weights,
                    # including valid pairs without conflict.
                    conflict_weight_sum = conflict_weight_sum + pair_w.sum()
                else:
                    # Original v2 ELC retained verbatim for ablation.
                    route_penalty = F.relu(route_sim - float(routing_margin))
                    weight = conflict * valid.to(conflict.dtype)
                    numerator = numerator + (weight * route_penalty).sum()
                    conflict_weight_sum = conflict_weight_sum + weight.sum()
                valid_pair_count = valid_pair_count + valid.sum().to(valid_pair_count.dtype)
                conflict_pair_count = conflict_pair_count + conflict_mask.sum().to(conflict_pair_count.dtype)
                valid_grad_cos_sum = valid_grad_cos_sum + (grad_cos * valid.to(grad_cos.dtype)).sum()
                conflict_route_sum = conflict_route_sum + (route_sim * conflict_mask.to(route_sim.dtype)).sum()

        if float(conflict_weight_sum.detach().item()) > 0.0:
            loss = numerator / conflict_weight_sum.clamp_min(1e-12)
        else:
            loss = soft_gate.sum() * 0.0

        vp = float(valid_pair_count.detach().item())
        cp = float(conflict_pair_count.detach().item())
        stats = {
            'valid_pairs': vp,
            'conflict_pairs': cp,
            'conflict_ratio': cp / max(vp, 1.0),
            'route_similarity': float((conflict_route_sum / conflict_pair_count.clamp_min(1.0)).detach().item()) if cp > 0 else 0.0,
            'gradient_cosine': float((valid_grad_cos_sum / valid_pair_count.clamp_min(1.0)).detach().item()) if vp > 0 else 0.0,
        }
        if use_gradient_density:
            stats['expert_overlap_conflict'] = float((density_expert_overlap_sum /
                conflict_pair_count.clamp_min(1.0)).detach().item()) if cp > 0 else 0.0
            stats['active_expert_margin_ratio'] = float((density_active_pair_count /
                conflict_pair_count.clamp_min(1.0)).detach().item()) if cp > 0 else 0.0
        stats.update(density_stats)
        return loss, stats

    def forward(self, z):
        # z: [B, M, D]
        # A detached copy is used to recompute the dense soft gate for ELC.
        self.last_router_input = z.detach()
        self.last_density_projection_output = z if self.training else None
        z_norm = self.norm(z)
        gate_logits = self.gate(z_norm) / self.gate_temperature       # [B, M, K]
        gate_probs = torch.softmax(gate_logits, dim=-1)               # [B, M, K]
        routing_weights = self._build_topk_routing(gate_probs)        # [B, M, K]

        # True sparse Top-K dispatch: each expert is evaluated only on the
        # channel tokens that actually selected it. The dense soft gate is
        # retained separately for ELC/load-balancing supervision.
        B, M, D = z.shape
        z_flat = z.reshape(B * M, D)
        routing_flat = routing_weights.reshape(B * M, self.num_experts)
        mixed_flat = torch.zeros_like(z_flat)
        selected_indices = []

        for k, expert in enumerate(self.experts):
            weight_k = routing_flat[:, k]
            idx = torch.nonzero(weight_k.detach() > 0, as_tuple=False).squeeze(-1)
            selected_indices.append(idx.detach())
            if idx.numel() == 0:
                expert.last_input = None
                expert.last_preact = None
                expert.last_hidden = None
                expert.last_output = None
                continue

            expert_in = z_flat.index_select(0, idx)                    # [N_k, D]
            # Keep the MoE-level scale effective when changed between passes.
            expert.scale = self.scale
            expert_out = expert(expert_in)                             # [N_k, D], includes scaled residual
            selected_w = weight_k.index_select(0, idx).unsqueeze(-1)
            # Keep the identity branch's forward weights but not its gate gradients.
            # This preserves the original behavior for st_topk as well as hard_topk.
            expert_delta = expert_out - expert_in  # Already scaled inside the expert.
            weighted = expert_in * selected_w.detach() + expert_delta * selected_w
            mixed_flat = mixed_flat.index_add(0, idx, weighted)

        mixed = mixed_flat.view(B, M, D)
        self.last_selected_flat_indices = selected_indices
        z_out = mixed  # Residual already supplied by selected experts

        lb_loss, importance_loss, switch_loss = self._load_balance_loss(gate_probs, routing_weights)
        aux_loss = lb_loss

        self.last_gate_soft_weights = gate_probs.detach()
        self.last_routing_weights = routing_weights.detach()
        self.last_dispatch_mask = (routing_weights.detach() > 0)
        self.last_lb_loss = lb_loss.detach()
        self.last_lb_importance_loss = importance_loss.detach()
        self.last_lb_switch_loss = switch_loss.detach()

        return z_out, aux_loss, lb_loss


class Encoder_PCA(nn.Module):
    def __init__(
        self,
        input_dim,
        word_embedding,
        hidden_dim=768,
        num_heads=12,
        num_encoder_layers=1,
        use_prealign_moe=False,
        prealign_moe_num_experts=4,
        prealign_moe_hidden=128,
        prealign_moe_dropout=0.0,
        prealign_moe_scale=0.2,
        prealign_moe_top_k=1,
        prealign_moe_routing_type="st_topk",
        prealign_moe_gate_temperature=1.0,
        prealign_moe_lb_type="switch",
        prealign_moe_lb_soft_weight=1.0,
        prealign_moe_lb_hard_weight=1.0,
        prealign_moe_lb_eps=1e-8,
        use_proto_route_mi_loss=False,
        proto_route_mi_weight=1.0,
        proto_relation_channel_chunk_size=16,
        proto_route_mi_hidden=128,
        proto_route_mi_dropout=0.0,
        proto_route_mi_target="soft",
        use_proto_route_ib=False,
        proto_route_ib_weight=1e-3,
        proto_route_ib_sample=True,
    ):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.word_embedding = word_embedding.T

        self.current_epoch = 0
        self.total_train_epochs = 1

        # Only the losses used by the current method are kept:
        #   1) channel-level MoE load-balance loss
        #   2) route-MI distillation loss
        #   3) optional variational information bottleneck KL loss
        self.last_moe_lb_loss = None
        self.last_proto_route_mi_loss = None
        self.last_proto_route_mi_raw_loss = None
        self.last_proto_route_mi_weight = None
        self.last_proto_route_ce_loss = None
        self.last_proto_route_ib_loss = None

        self.use_proto_route_mi_loss = bool(use_proto_route_mi_loss)
        self.proto_route_mi_weight = float(proto_route_mi_weight)
        self.proto_route_mi_eps = 1e-8
        self.proto_relation_channel_chunk_size = max(1, int(proto_relation_channel_chunk_size))
        self.proto_route_mi_hidden = int(proto_route_mi_hidden)
        self.proto_route_mi_dropout = float(proto_route_mi_dropout)
        self.proto_route_mi_target = str(proto_route_mi_target).lower()
        if self.proto_route_mi_target not in {"soft", "hard"}:
            raise ValueError("proto_route_mi_target must be either 'soft' or 'hard'")

        self.use_proto_route_ib = bool(use_proto_route_ib)
        self.proto_route_ib_weight = float(proto_route_ib_weight)
        self.proto_route_ib_sample = bool(proto_route_ib_sample)

        self.use_prealign_moe = bool(use_prealign_moe)
        if self.use_prealign_moe:
            self.prealign_moe = ChannelHardMoE(
                hidden_dim=hidden_dim,
                num_experts=prealign_moe_num_experts,
                bottleneck_dim=prealign_moe_hidden,
                dropout=prealign_moe_dropout,
                scale=prealign_moe_scale,
                top_k=prealign_moe_top_k,
                routing_type=prealign_moe_routing_type,
                gate_temperature=prealign_moe_gate_temperature,
                lb_type=prealign_moe_lb_type,
                lb_soft_weight=prealign_moe_lb_soft_weight,
                lb_hard_weight=prealign_moe_lb_hard_weight,
                lb_eps=prealign_moe_lb_eps,
            )
        else:
            self.prealign_moe = None

        # q_phi(z|v). z has the same dimension as v, so route-MI and IB share
        # exactly the same route decoder p_psi(g|.).
        #   without IB: g_hat = decoder(v)
        #   with IB:    z ~ q_phi(z|v), g_hat = decoder(z)
        if self.use_proto_route_ib:
            self.proto_route_ib_encoder = nn.Linear(hidden_dim, 2 * hidden_dim)
        else:
            self.proto_route_ib_encoder = None

        self.proto_route_decoder = nn.Sequential(
            nn.Linear(hidden_dim, int(proto_route_mi_hidden)),
            nn.GELU(),
            nn.Dropout(float(proto_route_mi_dropout)),
            nn.Linear(int(proto_route_mi_hidden), int(prealign_moe_num_experts)),
        )

    def set_current_epoch(self, epoch: int, total_epochs: int = None):
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            self.total_train_epochs = max(1, int(total_epochs))

    def _proto_route_mi_schedule_weight(self):
        """Enable route-MI/IB only in the second half of training."""
        total = max(1, int(getattr(self, 'total_train_epochs', 1)))
        start_epoch = total // 2 + 1
        return 1.0 if int(self.current_epoch) >= start_epoch else 0.0

    def _encode_route_bottleneck(self, route_repr):
        """Encode route-distillation representation into bottleneck latent z.

        route_repr: [B, C, D]
            In the current version this is the cross-attention output that is
            directly sent to the LLM.
        Returns:
            z:       [B, C, D], same dimension as route_repr so the route decoder is shared.
            kl_loss: scalar, mean KL(q_phi(z|route_repr) || N(0,I))
        """
        if not self.use_proto_route_ib or self.proto_route_ib_encoder is None:
            return route_repr, route_repr.new_zeros(())

        stats = self.proto_route_ib_encoder(route_repr)
        mu, logvar = torch.chunk(stats, chunks=2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if self.training and self.proto_route_ib_sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu

        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_loss = kl.sum(dim=-1).mean() / max(1, int(mu.size(-1)))
        return z, kl_loss

    def _route_mi_loss(self, aligned_repr):
        """Route-MI distillation on the representation that is directly sent to the LLM.

        aligned_repr: [B, M, D]
            Output of semantic prototype cross-attention. This is the same
            representation that will be returned as the text/LLM-side token
            representation, rather than the indirect prototype-weighted vector
            a_{b,m}E.

        MoE routing target:
            g_{b,m} = prealign_moe.last_gate_soft_weights, if target='soft'
            g_{b,m} = prealign_moe.last_routing_weights,   if target='hard'

        Optional information bottleneck:
            without IB: g_hat = decoder(aligned_repr)
            with IB:    z ~ q_phi(z|aligned_repr), g_hat = decoder(z)

        Loss:
            L = CE(sg(g), p_psi(g|aligned_repr or z))
                + beta_IB * KL(q_phi(z|aligned_repr)||N(0,I))
        """
        if aligned_repr is None or self.prealign_moe is None:
            return self.word_embedding.new_zeros(())

        if self.proto_route_mi_target == "hard":
            route_target = getattr(self.prealign_moe, "last_routing_weights", None)
        else:
            route_target = getattr(self.prealign_moe, "last_gate_soft_weights", None)
        if route_target is None:
            return aligned_repr.new_zeros(())

        eps = self.proto_route_mi_eps
        B, M, D = aligned_repr.shape
        route_target = route_target.detach().to(device=aligned_repr.device, dtype=aligned_repr.dtype)
        route_target = route_target.clamp_min(eps)
        route_target = route_target / route_target.sum(dim=-1, keepdim=True).clamp_min(eps)

        if route_target.shape[0] != B or route_target.shape[1] != M:
            raise RuntimeError(
                f"Route-MI shape mismatch: aligned_repr has shape {tuple(aligned_repr.shape)}, "
                f"but route_target has shape {tuple(route_target.shape)}. "
                "Route-MI expects both tensors to be channel-level [B, M, *]."
            )

        chunk_size = max(1, min(int(self.proto_relation_channel_chunk_size), M))
        total_ce_loss = aligned_repr.new_zeros(())
        total_ib_loss = aligned_repr.new_zeros(())
        total_count = 0
        total_chunks = 0

        for start in range(0, M, chunk_size):
            end = min(M, start + chunk_size)

            # Directly use the cross-attention output token that will be sent
            # to the LLM, instead of reconstructing a semantic vector by
            # sem_vec = attn_weights @ proto_embed.
            route_repr = aligned_repr[:, start:end, :].float()
            route_student, ib_loss = self._encode_route_bottleneck(route_repr)

            logits = self.proto_route_decoder(route_student)
            log_prob = F.log_softmax(logits, dim=-1)

            target = route_target[:, start:end, :].float()
            ce = -torch.sum(target * log_prob, dim=-1)
            total_ce_loss = total_ce_loss + ce.sum()
            total_ib_loss = total_ib_loss + ib_loss
            total_count += ce.numel()
            total_chunks += 1

        ce_loss = total_ce_loss / max(total_count, 1)
        ib_loss = total_ib_loss / max(total_chunks, 1)
        total_loss = ce_loss + self.proto_route_ib_weight * ib_loss

        self.last_proto_route_ce_loss = ce_loss.detach()
        self.last_proto_route_ib_loss = ib_loss.detach()
        return total_loss

    def forward(self, x):
        B = x.shape[0]

        word_embedding = self.word_embedding
        if word_embedding.ndim == 2:
            word_embedding = word_embedding.unsqueeze(0).expand(B, -1, -1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0:1].expand(B, -1, -1)

        # Cache the input to W_in for implicit per-channel rank-1 gradients.
        if self.prealign_moe is not None and self.training:
            self.prealign_moe.last_density_projection_input = x.detach()
        x = self.linear(x)  # [B, M, D]

        moe_aux_loss = x.new_zeros(())
        moe_lb_loss = x.new_zeros(())
        proto_route_mi_loss = x.new_zeros(())
        proto_route_mi_raw_loss = x.new_zeros(())
        proto_route_mi_weight = x.new_zeros(())

        self.last_moe_lb_loss = moe_lb_loss
        self.last_proto_route_mi_loss = proto_route_mi_loss
        self.last_proto_route_mi_raw_loss = proto_route_mi_raw_loss
        self.last_proto_route_mi_weight = proto_route_mi_weight
        self.last_proto_route_ce_loss = x.new_zeros(())
        self.last_proto_route_ib_loss = x.new_zeros(())

        if self.use_prealign_moe:
            x, moe_aux_loss, moe_lb_loss = self.prealign_moe(x)
            self.last_moe_lb_loss = moe_lb_loss.detach()

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x_time = x

        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x, attn_weights = self.cross_attention(
            q, k, v,
            # Route-MI now uses the cross-attention output x directly, so
            # attention weights are no longer needed for the auxiliary loss.
            need_weights=False,
            average_attn_weights=True,
        )

        if self.use_proto_route_mi_loss:
            schedule_weight = float(self._proto_route_mi_schedule_weight())
            proto_route_mi_weight = torch.tensor(schedule_weight, device=x.device, dtype=x.dtype)
            if schedule_weight > 0.0:
                aligned_repr = x.transpose(0, 1)  # [B, M, D], directly sent to the LLM after this layer
                proto_route_mi_raw_loss = self._route_mi_loss(aligned_repr)
                proto_route_mi_loss = proto_route_mi_raw_loss * proto_route_mi_weight
                moe_aux_loss = moe_aux_loss + self.proto_route_mi_weight * proto_route_mi_loss

            self.last_proto_route_mi_raw_loss = proto_route_mi_raw_loss.detach()
            self.last_proto_route_mi_weight = proto_route_mi_weight.detach()
            self.last_proto_route_mi_loss = proto_route_mi_loss.detach()

        x = x.transpose(0, 1)
        return x_time, x, moe_aux_loss, moe_lb_loss, proto_route_mi_loss


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        self.task_name = configs.task_name
        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)

        for _, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for _, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
        )
        self.text_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
        )

        self.in_layer = Encoder_PCA(
            configs.seq_len,
            word_embedding,
            hidden_dim=configs.d_model,
            use_prealign_moe=getattr(configs, 'use_prealign_moe', 0) == 1,
            prealign_moe_num_experts=getattr(configs, 'prealign_moe_num_experts', 4),
            prealign_moe_hidden=getattr(configs, 'prealign_moe_hidden', 128),
            prealign_moe_dropout=getattr(configs, 'prealign_moe_dropout', 0.0),
            prealign_moe_scale=getattr(configs, 'prealign_moe_scale', 0.2),
            prealign_moe_top_k=getattr(configs, 'prealign_moe_top_k', 1),
            prealign_moe_routing_type=getattr(configs, 'prealign_moe_routing_type', 'st_topk'),
            prealign_moe_gate_temperature=getattr(configs, 'prealign_moe_gate_temperature', 1.0),
            prealign_moe_lb_type=getattr(configs, 'prealign_moe_lb_type', 'switch'),
            prealign_moe_lb_soft_weight=getattr(configs, 'prealign_moe_lb_soft_weight', 1.0),
            prealign_moe_lb_hard_weight=getattr(configs, 'prealign_moe_lb_hard_weight', 1.0),
            prealign_moe_lb_eps=getattr(configs, 'prealign_moe_lb_eps', 1e-8),
            use_proto_route_mi_loss=getattr(configs, 'use_proto_route_mi_loss', 0) == 1,
            proto_route_mi_weight=getattr(configs, 'proto_route_mi_weight', 1.0),
            proto_relation_channel_chunk_size=getattr(configs, 'proto_relation_channel_chunk_size', 16),
            proto_route_mi_hidden=getattr(configs, 'proto_route_mi_hidden', 128),
            proto_route_mi_dropout=getattr(configs, 'proto_route_mi_dropout', 0.0),
            proto_route_mi_target=getattr(configs, 'proto_route_mi_target', 'soft'),
            use_proto_route_ib=getattr(configs, 'use_proto_route_ib', 0) == 1,
            proto_route_ib_weight=getattr(configs, 'proto_route_ib_weight', 1e-3),
            proto_route_ib_sample=getattr(configs, 'proto_route_ib_sample', 1) == 1,
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def set_current_epoch(self, epoch: int, total_epochs: int = None):
        if hasattr(self.in_layer, 'set_current_epoch'):
            self.in_layer.set_current_epoch(epoch, total_epochs=total_epochs)

    def expert_local_conflict_loss(
        self, probe_loss, conflict_margin=0.1, routing_margin=0.5,
        pair_shifts=1, min_grad_norm=1e-10,
        use_gradient_density=False, density_bandwidth=1.0,
        density_alpha=0.5, density_max_weight=5.0,
        density_pair_max_weight=5.0, density_routing_margin=0.01,
        density_channel_chunk_size=64,
    ):
        moe = getattr(self.in_layer, 'prealign_moe', None)
        if moe is None:
            return probe_loss * 0.0, {'valid_pairs': 0.0, 'conflict_pairs': 0.0,
                                     'conflict_ratio': 0.0, 'route_similarity': 0.0,
                                     'gradient_cosine': 0.0}
        return moe.expert_local_conflict_loss(
            probe_loss=probe_loss,
            conflict_margin=conflict_margin,
            routing_margin=routing_margin,
            pair_shifts=pair_shifts,
            min_grad_norm=min_grad_norm,
            use_gradient_density=use_gradient_density,
            density_bandwidth=density_bandwidth,
            density_alpha=density_alpha,
            density_max_weight=density_max_weight,
            density_pair_max_weight=density_pair_max_weight,
            density_routing_margin=density_routing_margin,
            density_channel_chunk_size=density_channel_chunk_size,
        )

    def forecast(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
        )
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
        )

        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'moe_aux_loss': moe_aux_loss,
            'moe_lb_loss': moe_lb_loss,
            'proto_route_mi_loss': proto_route_mi_loss,
            'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
            'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
            'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
            'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
        }

    def classification(self, x):
        B, L, M = x.shape
        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
        )
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
        )

        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'moe_aux_loss': moe_aux_loss,
            'moe_lb_loss': moe_lb_loss,
            'proto_route_mi_loss': proto_route_mi_loss,
            'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
            'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
            'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
            'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
        }

    def imputation(self, x, mask):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
        )
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
        )

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'moe_aux_loss': moe_aux_loss,
            'moe_lb_loss': moe_lb_loss,
            'proto_route_mi_loss': proto_route_mi_loss,
            'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
            'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
            'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
            'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss, moe_lb_loss, proto_route_mi_loss = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))]
        )
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))]
        )

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'moe_aux_loss': moe_aux_loss,
            'moe_lb_loss': moe_lb_loss,
            'proto_route_mi_loss': proto_route_mi_loss,
            'proto_route_mi_raw_loss': getattr(self.in_layer, 'last_proto_route_mi_raw_loss', proto_route_mi_loss),
            'proto_route_mi_weight': getattr(self.in_layer, 'last_proto_route_mi_weight', proto_route_mi_loss.new_zeros(())),
            'proto_route_ce_loss': getattr(self.in_layer, 'last_proto_route_ce_loss', proto_route_mi_loss),
            'proto_route_ib_loss': getattr(self.in_layer, 'last_proto_route_ib_loss', proto_route_mi_loss),
        }

    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        elif self.task_name == 'classification':
            output = self.classification(x)
        elif self.task_name == "imputation":
            output = self.imputation(x, mask)
        elif self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        else:
            output = self.forecast(x)
        return output

