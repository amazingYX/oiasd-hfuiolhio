# import argparse
# import os
# # Must be set before creating a CUDA context when deterministic cuBLAS is used.
# os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# import hashlib
# import json
# import sys
# import torch

# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
# from utils.print_args import print_args

# import random
# import numpy as np


# def set_reproducibility(seed, deterministic=True):
#     seed = int(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = bool(deterministic)
#     # Disable architecture-dependent TF32 rounding during reproduction runs.
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
#     torch.use_deterministic_algorithms(bool(deterministic))


# def file_sha256(path):
#     if not path or not os.path.isfile(path):
#         return None
#     dig = hashlib.sha256()
#     with open(path, 'rb') as stream:
#         for piece in iter(lambda: stream.read(1024 * 1024), b''):
#             dig.update(piece)
#     return dig.hexdigest()


# def write_repro_manifest(args, setting, seed):
#     import importlib.metadata as metadata
#     def package_version(pkg):
#         try:
#             return metadata.version(pkg)
#         except metadata.PackageNotFoundError:
#             return None
#     data_file = os.path.join(args.root_path, args.data_path)
#     report = {
#         'seed': seed,
#         'deterministic': bool(args.deterministic),
#         'python': sys.version,
#         'torch': torch.__version__,
#         'numpy': np.__version__,
#         'cuda_runtime': torch.version.cuda,
#         'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
#         'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if args.use_gpu else [],
#         'transformers': package_version('transformers'),
#         'peft': package_version('peft'),
#         'pandas': package_version('pandas'),
#         'scikit-learn': package_version('scikit-learn'),
#         'source_sha256': {name: file_sha256(name) for name in (
#             'run.py', 'models/CALF.py', 'utils/cmLoss.py',
#             'data_provider/data_factory.py', 'data_provider/data_loader.py',
#             'exp/exp_long_term_forecasting.py', 'exp/exp_basic.py')},
#         'CUBLAS_WORKSPACE_CONFIG': os.getenv('CUBLAS_WORKSPACE_CONFIG'),
#         'dataset_sha256': file_sha256(data_file),
#         'word_embedding_sha256': file_sha256(args.word_embedding_path),
#         'args': vars(args).copy(),
#     }
#     path = os.path.join(args.checkpoints, setting)
#     os.makedirs(path, exist_ok=True)
#     with open(os.path.join(path, 'repro_manifest.json'), 'w', encoding='utf-8') as out:
#         json.dump(report, out, ensure_ascii=False, indent=2, default=str)
#     print('[Repro] Seed={}, deterministic={}, dataset SHA256={}, CUDA={}, GPU={}'.format(
#         seed, bool(args.deterministic), report['dataset_sha256'],
#         report['cuda_runtime'], report['gpus']))


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='TimesNet')
#     parser.add_argument('--seed', type=int, default=2021,
#                         help='base seed; multiple itr runs use seed+run_index')
#     parser.add_argument('--deterministic', type=int, choices=[0, 1], default=1,
#                         help='strict PyTorch deterministic operations (1 may be slower or fail on unsupported GPU kernels)')

#     # basic config
#     parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
#                         help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
#     parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
#     parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
#     parser.add_argument('--model', type=str, required=True, default='Autoformer',
#                         help='model name, options: [Autoformer, Transformer, TimesNet]')

#     # data loader
#     parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#     parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#     parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

#     # imputation task
#     parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

#     # anomaly detection task
#     parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

#     # model define
#     parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
#     parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
#     parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=7, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false',
#                         help='whether to use distilling in encoder, using this argument means not using distilling',
#                         default=True)
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

#     # optimization
#     parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#     parser.add_argument('--task_loss', type=str, default='l1', help='task loss function')
#     parser.add_argument('--feature_loss', type=str, default='l1', help='distillation loss function')
#     parser.add_argument('--output_loss', type=str, default='l1', help='logits loss function')

#     # GPU
#     parser.add_argument('--use_gpu', type=lambda s: str(s).lower() in ('1', 'true', 'yes', 'y'),
#                         default=True, help='use gpu (true/false)')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

#     # de-stationary projector params
#     parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
#                         help='hidden layer dimensions of projector (List)')
#     parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

#     # self
#     parser.add_argument('--tmax', type=int, default=20)
#     parser.add_argument('--cos', type=int, default=1)

#     # lora
#     parser.add_argument('--r', type=int, default=8)
#     parser.add_argument('--lora_alpha', type=int, default=32)
#     parser.add_argument('--lora_dropout', type=float, default=0.1)

#     # align
#     parser.add_argument('--word_embedding_path', type=str, default="wte_pca_500.pt")

#     # loss weight
#     parser.add_argument('--task_w', type=float, default=1.0)
#     parser.add_argument('--feature_w', type=float, default=0.01)
#     parser.add_argument('--output_w', type=float, default=1.0)

#     # gpt
#     parser.add_argument('--gpt_layers', type=int, default=6, help='number of hidden layers in gpt')

#     # few shot percentage
#     parser.add_argument('--percent', type=int, default=100, help='few shot percentage')

#     # zero shot
#     parser.add_argument('--zero_shot', type=int, default=0, help='1 for training zero shot')
#     parser.add_argument('--target_data', type=str, default="ETTm1", help='target data for zero shot')

#     # channel-level hard Top-K MoE before cross-modal alignment
#     parser.add_argument('--use_prealign_moe', type=int, default=0, help='whether to enable channel-level hard Top-K pre-alignment MoE')
#     parser.add_argument('--prealign_moe_num_experts', type=int, default=4, help='number of pre-alignment experts')
#     parser.add_argument('--prealign_moe_hidden', type=int, default=128, help='bottleneck hidden size of each expert')
#     parser.add_argument('--prealign_moe_dropout', type=float, default=0.0, help='dropout in pre-alignment experts')
#     parser.add_argument('--prealign_moe_scale', type=float, default=0.2, help='residual scale for pre-alignment MoE')
#     parser.add_argument('--prealign_moe_top_k', type=int, default=1, help='number of experts selected for each channel token')
#     parser.add_argument('--prealign_moe_routing_type', type=str, default='st_topk', choices=['hard_topk', 'st_topk'], help='hard_topk uses detached Top-K routing; st_topk uses straight-through Top-K routing')
#     parser.add_argument('--prealign_moe_gate_temperature', type=float, default=1.0, help='temperature for channel-level gate softmax before Top-K')
#     parser.add_argument('--prealign_moe_lb_weight', type=float, default=1e-2, help='outer weight for MoE load-balance and alignment auxiliary losses')
#     parser.add_argument('--prealign_moe_lb_type', type=str, default='switch', choices=['importance', 'switch', 'combined'], help='load-balance loss type')
#     parser.add_argument('--prealign_moe_lb_soft_weight', type=float, default=1.0, help='inner weight for importance balance')
#     parser.add_argument('--prealign_moe_lb_hard_weight', type=float, default=1.0, help='inner weight for Switch-style dispatch balance')
#     parser.add_argument('--prealign_moe_lb_eps', type=float, default=1e-8, help='epsilon for load-balance loss')

#     # Expert-local gradient conflict routing regularization
#     parser.add_argument('--use_expert_local_conflict', type=int, default=0,
#                         help='whether to enable Expert-local gradient conflict regularization')
#     parser.add_argument('--expert_conflict_weight', type=float, default=1e-2,
#                         help='outer weight of Expert-local conflict loss')
#     parser.add_argument('--expert_conflict_start_epoch', type=int, default=3,
#                         help='1-indexed epoch at which Expert-local conflict regularization is activated')
#     parser.add_argument('--expert_conflict_margin', type=float, default=0.1,
#                         help='conflict margin tau_g: only gradient cosine < -tau_g is treated as harmful conflict')
#     parser.add_argument('--expert_routing_margin', type=float, default=0.5,
#                         help='routing cosine margin delta: no repulsion once soft-routing cosine <= delta')
#     parser.add_argument('--expert_conflict_pair_shifts', type=int, default=1,
#                         help='number of batch shifts used to sample same-channel token pairs for conflict estimation')
#     parser.add_argument('--expert_conflict_min_grad_norm', type=float, default=1e-10,
#                         help='ignore pair members whose full expert-gradient norm is below this threshold')

#     # Gradient-density-weighted expert-local conflict (DWGC). Requires ELC.
#     parser.add_argument('--use_gradient_density_weight', type=int, default=0,
#                         help='1: density-weighted expert-local conflict; 0: unchanged v2 ELC')
#     parser.add_argument('--gradient_density_tau', type=float, default=1.0,
#                         help='Gaussian kernel bandwidth for normalized W_in gradient directions')
#     parser.add_argument('--gradient_density_alpha', type=float, default=0.5,
#                         help='inverse-density exponent in [0,1], alpha=0 gives uniform weights')
#     parser.add_argument('--gradient_density_max_weight', type=float, default=5.0,
#                         help='upper cap on raw inverse-density sample weights')
#     parser.add_argument('--gradient_density_pair_max_weight', type=float, default=5.0,
#                         help='upper cap on geometric-mean pair weights')
#     parser.add_argument('--density_routing_margin', type=float, default=0.01,
#                         help='margin for per-expert soft probability product p_i,m * p_j,m; '
#                              'separate from full-distribution cosine margin in v2 ELC')
#     parser.add_argument('--density_channel_chunk_size', type=int, default=64,
#                         help='number of channels per exact gradient-density KDE chunk')

#     # route-MI + optional information bottleneck for cross-modal semantic alignment
#     parser.add_argument('--use_proto_route_mi_loss', type=int, default=0,
#                         help='whether to enable route-MI prototype auxiliary loss')
#     parser.add_argument('--proto_route_mi_weight', type=float, default=1.0,
#                         help='inner weight of route-MI/IB loss added into moe_aux_loss')
#     parser.add_argument('--proto_relation_channel_chunk_size', type=int, default=16,
#                         help='channel chunk size used by route-MI auxiliary loss to save memory')
#     parser.add_argument('--proto_route_mi_hidden', type=int, default=128,
#                         help='hidden dimension of route-MI decoder p_psi(g|v or z)')
#     parser.add_argument('--proto_route_mi_dropout', type=float, default=0.0,
#                         help='dropout in route-MI decoder')
#     parser.add_argument('--proto_route_mi_target', type=str, default='soft', choices=['soft', 'hard'],
#                         help='soft: distill dense MoE gate distribution; hard: distill actual Top-K routing distribution')
#     parser.add_argument(
#         '--proto_route_mi_start_epoch',
#         type=int,
#         default=-1,
#         help=(
#             '1-indexed epoch at which route-MI/IB distillation is enabled. '
#             'Set a positive integer to use an explicit start epoch; '
#             'values <= 0 keep the default behavior and start in the second half of training.'
#         ),
#     )

#     parser.add_argument('--use_proto_route_ib', type=int, default=0,
#                         help='whether to enable information bottleneck before route-MI decoder')
#     parser.add_argument('--proto_route_ib_weight', type=float, default=1e-3,
#                         help='weight beta for KL(q_phi(z|v)||N(0,I))')
#     parser.add_argument('--proto_route_ib_sample', type=int, default=1,
#                         help='1: sample z with reparameterization during training; 0: use mu only')


#     args = parser.parse_args()
#     args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#     set_reproducibility(args.seed, args.deterministic)

#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ', '')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]

#     print('Args in experiment:')
#     print_args(args)

#     if args.task_name == 'long_term_forecast':
#         Exp = Exp_Long_Term_Forecast
#     elif args.task_name == 'short_term_forecast':
#         Exp = Exp_Short_Term_Forecast
#     elif args.task_name == 'imputation':
#         Exp = Exp_Imputation
#     elif args.task_name == 'anomaly_detection':
#         Exp = Exp_Anomaly_Detection
#     elif args.task_name == 'classification':
#         Exp = Exp_Classification
#     else:
#         Exp = Exp_Long_Term_Forecast

#     moe_tag = ""
#     if args.use_prealign_moe:
#         moe_tag = (
#             f"_chmoe{args.use_prealign_moe}"
#             f"_e{args.prealign_moe_num_experts}"
#             f"_top{args.prealign_moe_top_k}"
#             f"_s{args.prealign_moe_scale}"
#             f"_rt{args.prealign_moe_routing_type}"
#             f"_lb{args.prealign_moe_lb_type}"
#         )
#         if args.use_expert_local_conflict:
#             moe_tag += (
#                 f"_elc1"
#                 f"_elcw{args.expert_conflict_weight}"
#                 f"_elcs{args.expert_conflict_start_epoch}"
#                 f"_cm{args.expert_conflict_margin}"
#                 f"_rm{args.expert_routing_margin}"
#             )
#             if args.use_gradient_density_weight:
#                 moe_tag += (
#                     f"_dwgc1_a{args.gradient_density_alpha}"
#                     f"_t{args.gradient_density_tau}"
#                     f"_wm{args.gradient_density_max_weight}"
#                     f"_pw{args.gradient_density_pair_max_weight}"
#                     f"_drm{args.density_routing_margin}"
#                 )
#         if args.use_proto_route_mi_loss:
#             moe_tag += (
#                 f"_rmi{args.use_proto_route_mi_loss}"
#                 f"_rmiw{args.proto_route_mi_weight}"
#                 f"_rt{args.proto_route_mi_target}"
#                 f"_rh{args.proto_route_mi_hidden}"
#                 f"_rd{args.proto_route_mi_dropout}"
#                 f"_rstart{args.proto_route_mi_start_epoch if args.proto_route_mi_start_epoch > 0 else 'half'}"
#             )
#             if args.use_proto_route_ib:
#                 moe_tag += (
#                     f"_ib1"
#                     f"_ibw{args.proto_route_ib_weight}"
#                     f"_ibs{args.proto_route_ib_sample}"
#                 )

#     if args.is_training:
#         base_seed = args.seed
#         for ii in range(args.itr):
#             run_seed = base_seed + ii
#             args.seed = run_seed  # used by per-split DataLoader generator
#             set_reproducibility(run_seed, args.deterministic)
#             exp = Exp(args)
#             setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_gpt{}_{}{}'.format(
#                 args.task_name,
#                 args.model_id,
#                 args.model,
#                 args.data,
#                 args.features,
#                 args.seq_len,
#                 args.label_len,
#                 args.pred_len,
#                 args.d_model,
#                 args.n_heads,
#                 args.e_layers,
#                 args.d_layers,
#                 args.d_ff,
#                 args.factor,
#                 args.embed,
#                 args.distil,
#                 args.des,
#                 args.gpt_layers,
#                 ii,
#                 moe_tag,
#             )

#             setting += f'_seed{run_seed}'
#             write_repro_manifest(args, setting, run_seed)
#             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             exp.train(setting)

#             print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#             exp.test(setting)
#             torch.cuda.empty_cache()
#     else:
#         ii = 0
#         set_reproducibility(args.seed, args.deterministic)
#         exp = Exp(args)
#         setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_gpt{}_{}{}'.format(
#             args.task_name,
#             args.model_id,
#             args.model,
#             args.data,
#             args.features,
#             args.seq_len,
#             args.label_len,
#             args.pred_len,
#             args.d_model,
#             args.n_heads,
#             args.e_layers,
#             args.d_layers,
#             args.d_ff,
#             args.factor,
#             args.embed,
#             args.distil,
#             args.des,
#             args.gpt_layers,
#             ii,
#             moe_tag,
#         )

#         setting += f'_seed{args.seed}'
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         exp.test(setting, test=1)
#         torch.cuda.empty_cache()


import argparse
import os
# Must be set before creating a CUDA context when deterministic cuBLAS is used.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args

import random
import numpy as np


def set_reproducibility(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    # Disable architecture-dependent TF32 rounding during reproduction runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(deterministic))


def file_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    dig = hashlib.sha256()
    with open(path, 'rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            dig.update(piece)
    return dig.hexdigest()


# Runtime-only flags do not change the learned model or the training configuration.
# Omitting them from the fingerprint lets train/test and different GPU choices
# resolve to the same experiment folder; every flag is still saved in run_args.json.
_RUN_ID_EXCLUDED = frozenset({
    'is_training', 'checkpoints', 'itr', 'gpu', 'use_gpu',
    'use_multi_gpu', 'devices', 'device_ids',
})


def fingerprint_config(args):
    """Return the canonical hyperparameter configuration and its 16-digit hash."""
    config = {k: v for k, v in vars(args).items() if k not in _RUN_ID_EXCLUDED}
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), default=str)
    return config, hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def safe_id_part(value, max_bytes=24):
    """ASCII-only bounded identifier, safe even for very long --model_id values."""
    part = re.sub(r'[^a-zA-Z0-9_-]+', '-', str(value)).strip('-_')
    return part[:max_bytes] or 'run'


def short_setting(args):
    """Short human-readable ID + fingerprint; never embeds all hyperparameters."""
    _, signature = fingerprint_config(args)
    task = {'long_term_forecast': 'LT', 'short_term_forecast': 'ST',
            'imputation': 'IMP', 'anomaly_detection': 'ANO',
            'classification': 'CLS'}.get(args.task_name, 'EXP')
    parts = [task, safe_id_part(args.model_id),
             f's{args.seq_len}', f'p{args.pred_len}']
    if args.use_prealign_moe:
        parts.append(f'M{args.prealign_moe_num_experts}K{args.prealign_moe_top_k}')
        if args.use_expert_local_conflict:
            parts.append('DWGC' if args.use_gradient_density_weight else 'ELC')
        if args.use_proto_route_mi_loss:
            parts.append('RGSA')
    if getattr(args, 'run_tag', ''):
        parts.append(safe_id_part(args.run_tag, 16))
    parts.extend([f'seed{args.seed}', signature])
    name = '_'.join(parts)
    assert len(name.encode('utf-8')) < 150, 'Unexpectedly long run ID'
    return name


def historical_setting(args, iteration_index):
    """Old verbose name is metadata only; never used as a directory name."""
    moe_tag = ''
    if args.use_prealign_moe:
        moe_tag = (
            f'_chmoe{args.use_prealign_moe}'
            f'_e{args.prealign_moe_num_experts}'
            f'_top{args.prealign_moe_top_k}'
            f'_s{args.prealign_moe_scale}'
            f'_rt{args.prealign_moe_routing_type}'
            f'_lb{args.prealign_moe_lb_type}'
        )
        if args.use_expert_local_conflict:
            moe_tag += (f'_elc1_elcw{args.expert_conflict_weight}'
                        f'_elcs{args.expert_conflict_start_epoch}'
                        f'_cm{args.expert_conflict_margin}'
                        f'_rm{args.expert_routing_margin}')
            if args.use_gradient_density_weight:
                moe_tag += (f'_dwgc1_a{args.gradient_density_alpha}'
                            f'_t{args.gradient_density_tau}'
                            f'_wm{args.gradient_density_max_weight}'
                            f'_pw{args.gradient_density_pair_max_weight}'
                            f'_drm{args.density_routing_margin}')
        if args.use_proto_route_mi_loss:
            moe_tag += (f'_rmi{args.use_proto_route_mi_loss}'
                        f'_rmiw{args.proto_route_mi_weight}'
                        f'_rt{args.proto_route_mi_target}'
                        f'_rh{args.proto_route_mi_hidden}'
                        f'_rd{args.proto_route_mi_dropout}'
                        f'_rstart{args.proto_route_mi_start_epoch if args.proto_route_mi_start_epoch > 0 else "half"}')
            if args.use_proto_route_ib:
                moe_tag += (f'_ib1_ibw{args.proto_route_ib_weight}'
                            f'_ibs{args.proto_route_ib_sample}')
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_gpt{}_{}{}_seed{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model,
        args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.distil, args.des, args.gpt_layers,
        iteration_index, moe_tag, args.seed,
    )


def save_json_atomic(path, data):
    """Avoid partially written configs on process interruption."""
    path = os.fspath(path)
    temp = path + '.tmp'
    with open(temp, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        out.write('\n')
    os.replace(temp, path)


def check_existing_config(path, expected_signature):
    """Never silently reuse/overwrite a checkpoint from another config."""
    manifest_path = os.path.join(path, 'repro_manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as inp:
            old = json.load(inp)
        existing = old.get('config_fingerprint')
        if existing is not None and existing != expected_signature:
            raise RuntimeError('Run-ID collision at {!r}: existing={} new={}'.format(
                path, existing, expected_signature))
    elif os.path.isdir(path) and os.listdir(path):
        raise RuntimeError('Nonempty run directory without a manifest: {!r}. '
                           'Choose another --model_id or move the directory first.'.format(path))


def export_run_metadata(args, setting):
    """Keep the full run configuration with evaluation outputs too."""
    checkpoint_dir = os.path.join(args.checkpoints, setting)
    for result_root in ('results', 'test_results'):
        destination = os.path.join('.', result_root, setting)
        if os.path.isdir(destination):
            for filename in ('run_args.json', 'repro_manifest.json'):
                source = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(destination, filename))


def collect_source_hashes():
    # Hash all repository Python sources, not just the model entry point.
    sources = [Path('run.py'), Path('pca.py')]
    for directory in ('models', 'exp', 'utils', 'data_provider'):
        if Path(directory).is_dir():
            sources.extend(sorted(Path(directory).rglob('*.py')))
    return {p.as_posix(): file_sha256(str(p)) for p in sources if p.is_file()}


def write_repro_manifest(args, setting, seed, iteration_index):
    import importlib.metadata as metadata
    def package_version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    data_file = os.path.join(args.root_path, args.data_path)
    report = {
        'seed': seed,
        'deterministic': bool(args.deterministic),
        'python': sys.version,
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if args.use_gpu else [],
        'transformers': package_version('transformers'),
        'peft': package_version('peft'),
        'pandas': package_version('pandas'),
        'scikit-learn': package_version('scikit-learn'),
        'source_sha256': collect_source_hashes(),
        'CUBLAS_WORKSPACE_CONFIG': os.getenv('CUBLAS_WORKSPACE_CONFIG'),
        'dataset_sha256': file_sha256(data_file),
        'word_embedding_sha256': file_sha256(args.word_embedding_path),
        'args': vars(args).copy(),
    }
    config, signature = fingerprint_config(args)
    report.update({
        'run_id': setting,
        'config_fingerprint': signature,
        'config_for_fingerprint': config,
        'historical_verbose_setting': historical_setting(args, iteration_index),
        'iteration_index': iteration_index,
        'command_line': [sys.executable, sys.argv[0], *sys.argv[1:]],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    path = os.path.join(args.checkpoints, setting)
    check_existing_config(path, signature)
    os.makedirs(path, exist_ok=True)
    save_json_atomic(os.path.join(path, 'run_args.json'), vars(args).copy())
    save_json_atomic(os.path.join(path, 'repro_manifest.json'), report)
    print('[Repro] Seed={}, deterministic={}, dataset SHA256={}, CUDA={}, GPU={}'.format(
        seed, bool(args.deterministic), report['dataset_sha256'],
        report['cuda_runtime'], report['gpus']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--seed', type=int, default=2021,
                        help='base seed; multiple itr runs use seed+run_index')
    parser.add_argument('--deterministic', type=int, choices=[0, 1], default=1,
                        help='strict PyTorch deterministic operations (1 may be slower or fail on unsupported GPU kernels)')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--run_tag', type=str, default='',
                        help='optional short tag to keep repeated trials separate (e.g. repeatA)')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--task_loss', type=str, default='l1', help='task loss function')
    parser.add_argument('--feature_loss', type=str, default='l1', help='distillation loss function')
    parser.add_argument('--output_loss', type=str, default='l1', help='logits loss function')

    # GPU
    parser.add_argument('--use_gpu', type=lambda s: str(s).lower() in ('1', 'true', 'yes', 'y'),
                        default=True, help='use gpu (true/false)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # self
    parser.add_argument('--tmax', type=int, default=20)
    parser.add_argument('--cos', type=int, default=1)

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # align
    parser.add_argument('--word_embedding_path', type=str, default="wte_pca_500.pt")

    # loss weight
    parser.add_argument('--task_w', type=float, default=1.0)
    parser.add_argument('--feature_w', type=float, default=0.01)
    parser.add_argument('--output_w', type=float, default=1.0)

    # gpt
    parser.add_argument('--gpt_layers', type=int, default=6, help='number of hidden layers in gpt')

    # few shot percentage
    parser.add_argument('--percent', type=int, default=100, help='few shot percentage')

    # zero shot
    parser.add_argument('--zero_shot', type=int, default=0, help='1 for training zero shot')
    parser.add_argument('--target_data', type=str, default="ETTm1", help='target data for zero shot')

    # channel-level hard Top-K MoE before cross-modal alignment
    parser.add_argument('--use_prealign_moe', type=int, default=0, help='whether to enable channel-level hard Top-K pre-alignment MoE')
    parser.add_argument('--prealign_moe_num_experts', type=int, default=4, help='number of pre-alignment experts')
    parser.add_argument('--prealign_moe_hidden', type=int, default=128, help='bottleneck hidden size of each expert')
    parser.add_argument('--prealign_moe_dropout', type=float, default=0.0, help='dropout in pre-alignment experts')
    parser.add_argument('--prealign_moe_scale', type=float, default=0.2, help='residual scale for pre-alignment MoE')
    parser.add_argument('--prealign_moe_top_k', type=int, default=1, help='number of experts selected for each channel token')
    parser.add_argument('--prealign_moe_routing_type', type=str, default='st_topk', choices=['hard_topk', 'st_topk'], help='hard_topk uses detached Top-K routing; st_topk uses straight-through Top-K routing')
    parser.add_argument('--prealign_moe_gate_temperature', type=float, default=1.0, help='temperature for channel-level gate softmax before Top-K')
    parser.add_argument('--prealign_moe_lb_weight', type=float, default=1e-2, help='outer weight for MoE load-balance and alignment auxiliary losses')
    parser.add_argument('--prealign_moe_lb_type', type=str, default='switch', choices=['importance', 'switch', 'combined'], help='load-balance loss type')
    parser.add_argument('--prealign_moe_lb_soft_weight', type=float, default=1.0, help='inner weight for importance balance')
    parser.add_argument('--prealign_moe_lb_hard_weight', type=float, default=1.0, help='inner weight for Switch-style dispatch balance')
    parser.add_argument('--prealign_moe_lb_eps', type=float, default=1e-8, help='epsilon for load-balance loss')

    # Expert-local gradient conflict routing regularization
    parser.add_argument('--use_expert_local_conflict', type=int, default=0,
                        help='whether to enable Expert-local gradient conflict regularization')
    parser.add_argument('--expert_conflict_weight', type=float, default=1e-2,
                        help='outer weight of Expert-local conflict loss')
    parser.add_argument('--expert_conflict_start_epoch', type=int, default=3,
                        help='1-indexed epoch at which Expert-local conflict regularization is activated')
    parser.add_argument('--expert_conflict_margin', type=float, default=0.1,
                        help='conflict margin tau_g: only gradient cosine < -tau_g is treated as harmful conflict')
    parser.add_argument('--expert_routing_margin', type=float, default=0.5,
                        help='routing cosine margin delta: no repulsion once soft-routing cosine <= delta')
    parser.add_argument('--expert_conflict_pair_shifts', type=int, default=1,
                        help='number of batch shifts used to sample same-channel token pairs for conflict estimation')
    parser.add_argument('--expert_conflict_min_grad_norm', type=float, default=1e-10,
                        help='ignore pair members whose full expert-gradient norm is below this threshold')

    # Gradient-density-weighted expert-local conflict (DWGC). Requires ELC.
    parser.add_argument('--use_gradient_density_weight', type=int, default=0,
                        help='1: density-weighted expert-local conflict; 0: unchanged v2 ELC')
    parser.add_argument('--gradient_density_tau', type=float, default=1.0,
                        help='Gaussian kernel bandwidth for normalized W_in gradient directions')
    parser.add_argument('--gradient_density_alpha', type=float, default=0.5,
                        help='inverse-density exponent in [0,1], alpha=0 gives uniform weights')
    parser.add_argument('--gradient_density_max_weight', type=float, default=5.0,
                        help='upper cap on raw inverse-density sample weights')
    parser.add_argument('--gradient_density_pair_max_weight', type=float, default=5.0,
                        help='upper cap on geometric-mean pair weights')
    parser.add_argument('--density_routing_margin', type=float, default=0.01,
                        help='margin for per-expert soft probability product p_i,m * p_j,m; '
                             'separate from full-distribution cosine margin in v2 ELC')
    parser.add_argument('--density_channel_chunk_size', type=int, default=64,
                        help='number of channels per exact gradient-density KDE chunk')

    # route-MI + optional information bottleneck for cross-modal semantic alignment
    parser.add_argument('--use_proto_route_mi_loss', type=int, default=0,
                        help='whether to enable route-MI prototype auxiliary loss')
    parser.add_argument('--proto_route_mi_weight', type=float, default=1.0,
                        help='inner weight of route-MI/IB loss added into moe_aux_loss')
    parser.add_argument('--proto_relation_channel_chunk_size', type=int, default=16,
                        help='channel chunk size used by route-MI auxiliary loss to save memory')
    parser.add_argument('--proto_route_mi_hidden', type=int, default=128,
                        help='hidden dimension of route-MI decoder p_psi(g|v or z)')
    parser.add_argument('--proto_route_mi_dropout', type=float, default=0.0,
                        help='dropout in route-MI decoder')
    parser.add_argument('--proto_route_mi_target', type=str, default='soft', choices=['soft', 'hard'],
                        help='soft: distill dense MoE gate distribution; hard: distill actual Top-K routing distribution')
    parser.add_argument(
        '--proto_route_mi_start_epoch',
        type=int,
        default=-1,
        help=(
            '1-indexed epoch at which route-MI/IB distillation is enabled. '
            'Set a positive integer to use an explicit start epoch; '
            'values <= 0 keep the default behavior and start in the second half of training.'
        ),
    )

    parser.add_argument('--use_proto_route_ib', type=int, default=0,
                        help='whether to enable information bottleneck before route-MI decoder')
    parser.add_argument('--proto_route_ib_weight', type=float, default=1e-3,
                        help='weight beta for KL(q_phi(z|v)||N(0,I))')
    parser.add_argument('--proto_route_ib_sample', type=int, default=1,
                        help='1: sample z with reparameterization during training; 0: use mu only')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    set_reproducibility(args.seed, args.deterministic)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        base_seed = args.seed
        for ii in range(args.itr):
            run_seed = base_seed + ii
            args.seed = run_seed  # also seeds per-split DataLoader generators
            set_reproducibility(run_seed, args.deterministic)
            exp = Exp(args)
            setting = short_setting(args)
            write_repro_manifest(args, setting, run_seed, ii)
            print('[Run ID] {} ({} bytes)'.format(setting, len(setting.encode('utf-8'))))
            print('[Run config] {}/{}/run_args.json'.format(args.checkpoints.rstrip('/'), setting))
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            export_run_metadata(args, setting)
            torch.cuda.empty_cache()
    else:
        set_reproducibility(args.seed, args.deterministic)
        exp = Exp(args)
        setting = short_setting(args)
        checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            # Allow inference with checkpoints created by the original naming scheme.
            old_name = historical_setting(args, 0)
            old_path = os.path.join(args.checkpoints, old_name, 'checkpoint.pth')
            if len(old_name.encode('utf-8')) <= 255 and os.path.isfile(old_path):
                print('[Run ID] Found legacy checkpoint; using its original directory.')
                setting = old_name
            else:
                raise FileNotFoundError('Checkpoint not found: {}. To test a previous run, '
                                        'pass the same training hyperparameters and seed. '
                                        'Full settings are recorded in checkpoint/run_args.json.'.format(checkpoint_path))
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        export_run_metadata(args, setting)
        torch.cuda.empty_cache()
