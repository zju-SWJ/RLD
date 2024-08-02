import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def swap(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    gt_index = target.unsqueeze(1)
    max_value, max_index = torch.max(logits, dim=1, keepdim=True)
    swapped_logits = copy.deepcopy(logits).scatter_(1, max_index, gt_value).scatter_(1, gt_index, max_value)
    return swapped_logits

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class SWAP(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(SWAP, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            logits_teacher = swap(logits_teacher, target)

        # losses
        loss_kd = kd_loss(logits_student, logits_teacher, self.temperature, self.logit_stand)
        losses_dict = {
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
