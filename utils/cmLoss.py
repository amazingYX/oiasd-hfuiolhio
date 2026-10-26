import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class cmLoss(nn.Module):
    def __init__(self, feature_loss, output_loss, task_loss, task_name, feature_w=0.01, output_w=1.0, task_w=1.0):
        super(cmLoss, self).__init__()
        self.task_w = task_w
        self.output_w = output_w
        self.feature_w = feature_w

        self.feature_loss = loss_dict[feature_loss]
        self.output_loss = loss_dict[output_loss]
        self.task_loss = loss_dict[task_loss]
        
        self.task_name = task_name

    def compute_task_loss(self, outputs_time, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        """Compute exactly the supervised task/forecasting objective used by cmLoss.

        ELC calls this method to construct its gradient-conflict probe, so the
        gradient geometry is defined by the same forecasting loss optimized by
        the main training objective. Feature/output alignment losses are
        intentionally excluded from the probe.
        """
        batch_y = batch_y.to(outputs_time.device)

        if self.task_name == "long_term_forecast":
            return self.task_loss(outputs_time, batch_y)
        elif self.task_name == "short_term_forecast":
            return self.task_loss(
                in_sample, freq_map, outputs_time, batch_y, batch_y_mark
            )
        elif self.task_name == "classification":
            return self.task_loss(outputs_time, batch_y)
        elif self.task_name == "imputation":
            return self.task_loss(outputs_time, batch_y)
        elif self.task_name == "anomaly_detection":
            return self.task_loss(outputs_time, batch_y)
        raise ValueError(f"Unsupported task_name for task loss: {self.task_name}")

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"],
        )
        
        # feture regularization loss
        feature_loss = sum(
            [
                (0.8**idx) * self.feature_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                    zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
                )
            ]
        )

        # output consistency loss
        if self.task_name == "long_term_forecast":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "short_term_forecast":
            output_loss = self.output_loss(in_sample, freq_map, outputs_time, outputs_text, batch_y_mark)
        elif self.task_name == "classification":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "imputation":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "anomaly_detection":
            output_loss = self.output_loss(outputs_time, outputs_text)
            

        batch_y = batch_y.to(output_loss.device)

        # Supervised task loss. Keep this in one helper so ELC can reuse the
        # exact same forecasting objective rather than a hard-coded proxy.
        task_loss = self.compute_task_loss(
            outputs_time, batch_y, in_sample, freq_map, batch_y_mark
        )

        total_loss = self.task_w * task_loss + self.output_w * output_loss + self.feature_w * feature_loss
        return total_loss