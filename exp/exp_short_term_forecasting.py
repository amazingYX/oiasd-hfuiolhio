# from data_provider.data_factory import data_provider
# from data_provider.m4 import M4Meta
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.losses import mape_loss, mase_loss, smape_loss
# from utils.m4_summary import M4Summary
# from utils.cmLoss import cmLoss
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import numpy as np
# import pandas

# warnings.filterwarnings('ignore')


# class Exp_Short_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Short_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         if self.args.data == 'm4':
#             self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
#             self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
#             self.args.label_len = self.args.pred_len
#             self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
#         model = self.model_dict[self.args.model].Model(self.args, self.device).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         param_dict = [
#             {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
#             {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
#         ]
#         model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
#         loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

#         return model_optim, loss_optim

#     def _select_criterion(self):
#         criterion = cmLoss(self.args.feature_loss, 
#                            self.args.output_loss, 
#                            self.args.task_loss, 
#                            self.args.task_name, 
#                            self.args.feature_w, 
#                            self.args.output_w, 
#                            self.args.task_w)
#         return criterion
    
#     def _select_vali_criterion(self, task_loss):
#         if task_loss == 'mape':
#             criterion = mape_loss()
#         elif task_loss == 'mase':
#             criterion = mase_loss()
#         elif task_loss == 'smape':
#             criterion = smape_loss()
#         return criterion

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim, loss_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 loss_optim.zero_grad()

#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 outputs_dict = self.model(batch_x)

#                 batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
#                 batch_y_mark = batch_y_mark[:, -self.args.pred_len:].to(self.device)
                
#                 loss_value = criterion(outputs_dict, batch_y, batch_x, self.args.frequency_map, batch_y_mark)

#                 loss = loss_value  # + loss_sharpness * 1e-5
#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 loss.backward()
#                 model_optim.step()
#                 loss_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(train_loader, vali_loader, self._select_vali_criterion("smape"))
#             test_loss = vali_loss
#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
#             if self.args.cos:
#                 scheduler.step()
#                 print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
#             else:
#                 adjust_learning_rate(model_optim, epoch + 1, self.args)
            
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def vali(self, train_loader, vali_loader, criterion):
#         x, _ = train_loader.dataset.last_insample_window()
#         y = vali_loader.dataset.timeseries
#         x = torch.tensor(x, dtype=torch.float32).to(self.device)
#         x = x.unsqueeze(-1)

#         self.model.in_layer.eval()
#         self.model.out_layer.eval()
#         self.model.time_proj.eval()
#         self.model.text_proj.eval()

#         with torch.no_grad():
#             # decoder input
#             B, _, C = x.shape

#             # encoder - decoder
#             outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
#             id_list = np.arange(0, B, 500)  # validation set size
#             id_list = np.append(id_list, B)

#             for i in range(len(id_list) - 1):
#                 outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]])["outputs_time"].detach().cpu()

#             pred = outputs
#             true = torch.from_numpy(np.array(y))
#             batch_y_mark = torch.ones(true.shape)

#             loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

#         self.model.in_layer.train()
#         self.model.out_layer.train()
#         self.model.time_proj.train()
#         self.model.text_proj.train()
#         return loss

#     def test(self, setting, test=0):
#         _, train_loader = self._get_data(flag='train')
#         _, test_loader = self._get_data(flag='test')
#         x, _ = train_loader.dataset.last_insample_window()
#         y = test_loader.dataset.timeseries
#         x = torch.tensor(x, dtype=torch.float32).to(self.device)
#         x = x.unsqueeze(-1)

#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             B, _, C = x.shape

#             # encoder - decoder
#             outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
#             id_list = np.arange(0, B, 500)
#             id_list = np.append(id_list, B)
#             for i in range(len(id_list) - 1):
#                 outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]])["outputs_time"]

#                 if id_list[i] % 1000 == 0:
#                     print(id_list[i])

#             f_dim = -1 if self.args.features == 'MS' else 0
#             outputs = outputs[:, -self.args.pred_len:, f_dim:]
#             outputs = outputs.detach().cpu().numpy()

#             preds = outputs
#             trues = y
#             x = x.detach().cpu().numpy()

#             for i in range(0, preds.shape[0], preds.shape[0] // 10):
#                 gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
#                 pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#         print('test shape:', preds.shape)

#         # result save
#         folder_path = './m4_results/' + self.args.model + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
#         forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
#         forecasts_df.index.name = 'id'
#         forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
#         forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

#         print(self.args.model)
#         file_path = './m4_results/' + self.args.model + '/'
#         if 'Weekly_forecast.csv' in os.listdir(file_path) \
#                 and 'Monthly_forecast.csv' in os.listdir(file_path) \
#                 and 'Yearly_forecast.csv' in os.listdir(file_path) \
#                 and 'Daily_forecast.csv' in os.listdir(file_path) \
#                 and 'Hourly_forecast.csv' in os.listdir(file_path) \
#                 and 'Quarterly_forecast.csv' in os.listdir(file_path):
#             m4_summary = M4Summary(file_path, self.args.root_path)
#             # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
#             smape_results, owa_results, mape, mase = m4_summary.evaluate()
#             print('smape:', smape_results)
#             print('mape:', mape)
#             print('mase:', mase)
#             print('owa:', owa_results)
#         else:
#             print('After all 6 tasks are finished, you can calculate the averaged index')
#         return
from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion
    
    def _select_vali_criterion(self, task_loss):
        if task_loss == 'mape':
            criterion = mape_loss()
        elif task_loss == 'mase':
            criterion = mase_loss()
        elif task_loss == 'smape':
            criterion = smape_loss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.tmax, eta_min=1e-8
        )

        # Outer coefficient for all MoE auxiliary terms. In the current CALF
        # implementation, moe_aux_loss contains the MoE load-balance term and,
        # after activation, the weighted Route-MI/IB distillation term.
        moe_lb_weight = float(getattr(self.args, 'prealign_moe_lb_weight', 0.0))

        # Expert-local conflict (ELC) configuration.
        elc_enabled = bool(getattr(self.args, 'use_expert_local_conflict', 0))
        elc_weight = float(getattr(self.args, 'expert_conflict_weight', 1e-2))
        elc_start_epoch = max(1, int(getattr(self.args, 'expert_conflict_start_epoch', 3)))
        elc_conflict_margin = float(getattr(self.args, 'expert_conflict_margin', 0.1))
        elc_routing_margin = float(getattr(self.args, 'expert_routing_margin', 0.5))
        elc_pair_shifts = max(1, int(getattr(self.args, 'expert_conflict_pair_shifts', 1)))
        elc_min_grad_norm = float(getattr(self.args, 'expert_conflict_min_grad_norm', 1e-10))
        density_enabled = bool(getattr(self.args, 'use_gradient_density_weight', 0))
        density_bandwidth = float(getattr(self.args, 'gradient_density_tau', 1.0))
        density_alpha = float(getattr(self.args, 'gradient_density_alpha', 0.5))
        density_max_weight = float(getattr(self.args, 'gradient_density_max_weight', 5.0))
        density_pair_max_weight = float(getattr(self.args, 'gradient_density_pair_max_weight', 5.0))
        density_routing_margin = float(getattr(self.args, 'density_routing_margin', 0.01))
        density_channel_chunk_size = int(getattr(self.args, 'density_channel_chunk_size', 64))
        if density_enabled and not elc_enabled:
            raise ValueError('DWGC requires --use_expert_local_conflict 1')

        if elc_enabled:
            print(
                '[ELC schedule] start_epoch={}, weight={}, conflict_margin={}, '
                'routing_margin={}, pair_shifts={}'.format(
                    elc_start_epoch, elc_weight, elc_conflict_margin,
                    elc_routing_margin, elc_pair_shifts
                )
            )
            if density_enabled:
                print('[DWGC] tau={}, alpha={}, max_weight={}, pair_max={}, expert_margin={}, channel_chunk={}'.format(
                    density_bandwidth, density_alpha, density_max_weight,
                    density_pair_max_weight, density_routing_margin, density_channel_chunk_size))

        # Route-MI/IB distillation schedule.
        # run.py already defines --proto_route_mi_start_epoch as a 1-indexed
        # epoch. A positive value explicitly sets the start epoch; values <= 0
        # preserve the original behavior and start in the second half.
        distillation_enabled = bool(getattr(self.args, 'use_proto_route_mi_loss', 0))
        configured_start_epoch = int(getattr(self.args, 'proto_route_mi_start_epoch', -1))

        if configured_start_epoch > 0:
            distill_start_epoch = configured_start_epoch

            # CALF internally activates Route-MI at:
            #     internal_start = total_epochs // 2 + 1
            # Passing this effective total produces exactly the requested
            # 1-indexed start epoch without changing the real training length.
            schedule_total_epochs = max(1, 2 * (distill_start_epoch - 1))
            schedule_mode = 'explicit'
        else:
            distill_start_epoch = self.args.train_epochs // 2 + 1
            schedule_total_epochs = self.args.train_epochs
            schedule_mode = 'default-half'

        model_for_epoch = self.model.module if hasattr(self.model, 'module') else self.model
        has_epoch_api = hasattr(model_for_epoch, 'set_current_epoch')

        if distillation_enabled:
            print(
                '[Route-MI schedule] mode={}, start_epoch={}, train_epochs={}'.format(
                    schedule_mode, distill_start_epoch, self.args.train_epochs
                )
            )
            if not has_epoch_api:
                print(
                    '[Route-MI schedule] Warning: the model has no set_current_epoch() method; '
                    'the requested start epoch cannot be propagated to the model.'
                )
            if distill_start_epoch > self.args.train_epochs:
                print(
                    '[Route-MI schedule] Warning: start_epoch={} exceeds train_epochs={}; '
                    'the distillation branch will not be activated.'.format(
                        distill_start_epoch, self.args.train_epochs
                    )
                )
        else:
            print('[Route-MI schedule] disabled because use_proto_route_mi_loss=0.')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            # Inform CALF of the current 1-indexed epoch before any forward pass.
            # For explicit scheduling, schedule_total_epochs is an effective
            # value chosen to make CALF's internal half-training switch activate
            # exactly at distill_start_epoch.
            cur_epoch = epoch + 1
            if has_epoch_api:
                model_for_epoch.set_current_epoch(
                    cur_epoch,
                    total_epochs=schedule_total_epochs,
                )

            if distillation_enabled and cur_epoch == distill_start_epoch:
                print(
                    '[Route-MI schedule] Distillation branch activated at epoch {}.'.format(
                        cur_epoch
                    )
                )
            if elc_enabled and cur_epoch == elc_start_epoch:
                print('[ELC schedule] Expert-local conflict regularization activated at epoch {}.'.format(cur_epoch))
                print('[ELC objective] conflict gradients use forecasting task_loss={} only.'.format(self.args.task_loss))

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs_dict = self.model(batch_x)

                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:].to(self.device)

                base_loss = criterion(
                    outputs_dict,
                    batch_y,
                    batch_x,
                    self.args.frequency_map,
                    batch_y_mark,
                )

                # cmLoss computes the original CALF/M4 objective. Route-MI/IB
                # and MoE load balancing are returned separately by CALF and
                # must be added here; otherwise the distillation branch can be
                # executed but will not influence optimization.
                moe_aux_loss = outputs_dict.get('moe_aux_loss', None)
                if moe_aux_loss is None:
                    moe_aux_loss = torch.zeros(
                        (), device=base_loss.device, dtype=base_loss.dtype
                    )

                elc_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
                elc_stats = {
                    'valid_pairs': 0.0,
                    'conflict_pairs': 0.0,
                    'conflict_ratio': 0.0,
                    'route_similarity': 0.0,
                    'gradient_cosine': 0.0,
                }
                if elc_enabled and cur_epoch >= elc_start_epoch:
                    probe_pred = outputs_dict['outputs_time']
                    probe_true = batch_y
                    if probe_pred.dim() == 3 and probe_true.dim() == 2:
                        probe_true = probe_true.unsqueeze(-1)
                    elif probe_pred.dim() == 2 and probe_true.dim() == 3 and probe_true.size(-1) == 1:
                        probe_true = probe_true.squeeze(-1)
                    probe_pred = probe_pred[:, -probe_true.size(1):]
                    # Use exactly the same forecasting task loss as the main
                    # short-term objective; do not include CALF feature/output
                    # alignment losses in the ELC gradient probe.
                    probe_loss = criterion.compute_task_loss(
                        probe_pred,
                        probe_true,
                        in_sample=batch_x,
                        freq_map=self.args.frequency_map,
                        batch_y_mark=batch_y_mark,
                    )
                    elc_loss, elc_stats = model_for_epoch.expert_local_conflict_loss(
                        probe_loss=probe_loss,
                        conflict_margin=elc_conflict_margin,
                        routing_margin=elc_routing_margin,
                        pair_shifts=elc_pair_shifts,
                        min_grad_norm=elc_min_grad_norm,
                        use_gradient_density=density_enabled,
                        density_bandwidth=density_bandwidth,
                        density_alpha=density_alpha,
                        density_max_weight=density_max_weight,
                        density_pair_max_weight=density_pair_max_weight,
                        density_routing_margin=density_routing_margin,
                        density_channel_chunk_size=density_channel_chunk_size,
                    )

                loss = base_loss + moe_lb_weight * moe_aux_loss + elc_weight * elc_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | base: {3:.7f} | "
                        "moe_aux: {4:.7f} | elc: {5:.7f} | conflict_ratio: {6:.4f} | "
                        "route_sim: {7:.4f} | grad_cos: {8:.4f} | pairs: {9:.0f}".format(
                            i + 1, cur_epoch, loss.item(), base_loss.item(), moe_aux_loss.item(),
                            elc_loss.item(), elc_stats['conflict_ratio'],
                            elc_stats['route_similarity'], elc_stats['gradient_cosine'],
                            elc_stats['valid_pairs']
                        )
                    )
                    if density_enabled and 'density_mean' in elc_stats:
                        print('\tDWGC | density: {0:.5f} | density_min: {1:.5f} | '
                              'density_max: {2:.5f} | sample_w: [{3:.3f}, {4:.3f}] | expert_overlap: {5:.5f} | active_margin: {6:.3f}'.format(
                                  elc_stats['density_mean'], elc_stats['density_min'],
                                  elc_stats['density_max'], elc_stats['sample_weight_min'],
                                  elc_stats['sample_weight_max'],
                                  elc_stats['expert_overlap_conflict'],
                                  elc_stats['active_expert_margin_ratio']))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                loss_optim.step()

            print("Epoch: {} cost time: {}".format(cur_epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, self._select_vali_criterion("smape"))
            test_loss = vali_loss
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    cur_epoch, train_steps, train_loss, vali_loss, test_loss
                )
            )

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, cur_epoch, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        was_training = self.model.training
        self.model.eval()  # includes both GPT-2 backbones and dropout

        with torch.no_grad():
            # decoder input
            B, _, C = x.shape

            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)

            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]])["outputs_time"].detach().cpu()

            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train(was_training)
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape

            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 500)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]])["outputs_time"]

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return