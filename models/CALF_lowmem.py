import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model

from models.GPT2_arch import AccustumGPT2Model


class ResidualExpert(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, M, D]
        return self.up(self.dropout(self.act(self.down(x))))


class PreAlignmentSoftMoE(nn.Module):
    """
    Lightweight soft MoE inserted before cross-modal alignment.

    Input:
        z: [B, M, D] temporal representations after transformer_encoder
    Output:
        z_out: [B, M, D] residual expert-shaped representations
        aux_loss: load-balancing auxiliary loss
    """
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        bottleneck_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 0.2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.scale = scale
        self.norm = nn.LayerNorm(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.experts = nn.ModuleList(
            [ResidualExpert(hidden_dim, bottleneck_dim, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, z):
        # z: [B, M, D]
        pooled = self.norm(z).mean(dim=1)                   # [B, D]
        gate_logits = self.gate(pooled)                     # [B, K]
        gate_weights = torch.softmax(gate_logits, dim=-1)   # [B, K]

        # Streaming weighted sum to reduce peak memory.
        # Avoid constructing a [B, K, M, D] tensor with torch.stack.
        mixed = torch.zeros_like(z)
        for k, expert in enumerate(self.experts):
            expert_out = expert(z)                          # [B, M, D]
            weight = gate_weights[:, k].view(-1, 1, 1)     # [B, 1, 1]
            mixed = mixed + weight * expert_out

        z_out = z + self.scale * mixed

        # Simple load-balance loss: smaller is better when usage is more uniform
        avg_usage = gate_weights.mean(dim=0)  # [K]
        aux_loss = self.num_experts * torch.sum(avg_usage ** 2)

        return z_out, aux_loss


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
    ):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.word_embedding = word_embedding.T

        self.use_prealign_moe = use_prealign_moe
        if self.use_prealign_moe:
            self.prealign_moe = PreAlignmentSoftMoE(
                hidden_dim=hidden_dim,
                num_experts=prealign_moe_num_experts,
                bottleneck_dim=prealign_moe_hidden,
                dropout=prealign_moe_dropout,
                scale=prealign_moe_scale,
            )
        else:
            self.prealign_moe = None

    def forward(self, x):
        B = x.shape[0]

        # Keep the original embedding tensor on the module and build a
        # batch-shaped local view only for the current forward pass.
        word_embedding = self.word_embedding
        if word_embedding.ndim == 2:
            word_embedding = word_embedding.unsqueeze(0).expand(B, -1, -1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0:1].expand(B, -1, -1)

        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        moe_aux_loss = x.new_zeros(())
        if self.use_prealign_moe:
            x, moe_aux_loss = self.prealign_moe(x)

        x_time = x
        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)
        x = x.transpose(0, 1)
        return x_time, x, moe_aux_loss


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

    def forecast(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss = self.in_layer(x)

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
        }

    def classification(self, x):
        B, L, M = x.shape
        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss = self.in_layer(x)

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
        }

    def imputation(self, x, mask):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss = self.in_layer(x)

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
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1, moe_aux_loss = self.in_layer(x)

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
