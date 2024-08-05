import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def rld_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand, alpha_temperature):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # scd loss
    student_gt_mask = _get_gt_mask(logits_student, target)
    student_other_mask = _get_other_mask(logits_student, target)
    max_index = torch.argmax(logits_teacher, dim=1)
    teacher_max_mask = _get_gt_mask(logits_teacher, max_index)
    teacher_other_mask = _get_other_mask(logits_teacher, max_index)
    pred_student = F.softmax(logits_student / alpha_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / alpha_temperature, dim=1)
    pred_student = cat_mask(pred_student, student_gt_mask, student_other_mask)
    pred_teacher = cat_mask(pred_teacher, teacher_max_mask, teacher_other_mask)
    log_pred_student = torch.log(pred_student)
    scd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (alpha_temperature**2)

    # mcd loss
    mask = _get_ge_mask(logits_teacher, target)
    assert mask.shape == logits_student.shape
    masked_student = (logits_student / temperature).masked_fill(mask, -1e9)
    log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
    masked_teacher = (logits_teacher / temperature).masked_fill(mask, -1e9)
    pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
    mcd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)

    return alpha * scd_loss + beta * mcd_loss


def _get_ge_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits >= gt_value, 1, 0).bool()
    return mask

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class RLD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(RLD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.RLD.CE_WEIGHT
        self.alpha = cfg.RLD.ALPHA
        self.beta = cfg.RLD.BETA
        self.temperature = cfg.RLD.T
        self.warmup = cfg.RLD.WARMUP
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.alpha_temperature = cfg.RLD.ALPHA_T

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_rld = min(kwargs["epoch"] / self.warmup, 1.0) * rld_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.logit_stand,
            self.alpha_temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_rld,
        }
        return logits_student, losses_dict
