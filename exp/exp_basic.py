import os
import torch
from models import CALF


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "CALF": CALF,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        # CUDA_VISIBLE_DEVICES remaps visible cards to cuda:0, cuda:1, ...
        # Respect --use_gpu and the user-supplied index within that visible set.
        if bool(getattr(self.args, 'use_gpu', False)) and torch.cuda.is_available():
            gpu_index = int(getattr(self.args, 'gpu', 0))
            if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
                raise ValueError(
                    f'GPU index {gpu_index} invalid for '
                    f'{torch.cuda.device_count()} visible CUDA device(s). '
                    'When using CUDA_VISIBLE_DEVICES, --gpu is the remapped index.'
                )
            device = torch.device(f'cuda:{gpu_index}')
            print(f'Use GPU: {device} ({torch.cuda.get_device_name(gpu_index)})')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
