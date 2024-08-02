import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def rc_loss(logits_student_in, target, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in

    max_value, _ = torch.max(logits_student, dim=1, keepdim=True)
    gt_value = torch.gather(logits_student, 1, target.unsqueeze(1))
    src = (gt_value + max_value).detach()
    logits_teacher = logits_student.detach().clone().scatter_(1, target.unsqueeze(1), src)

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_rc = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_rc *= temperature**2
    return loss_rc


class RC(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(RC, self).__init__(student, teacher)
        self.temperature = cfg.RC.T
        self.ce_loss_weight = cfg.RC.CE_WEIGHT
        self.kd_loss_weight = cfg.RC.KD_WEIGHT
        self.rc_loss_weight = cfg.RC.RC_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature, self.logit_stand)
        loss_rc = self.rc_loss_weight * rc_loss(logits_student, target, self.temperature, self.logit_stand)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_rc": loss_rc,
        }
        return logits_student, losses_dict
