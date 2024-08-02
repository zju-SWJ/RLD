import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def my_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand, method, loss_type, alpha_temperature):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # tckd-like loss
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
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (alpha_temperature**2)

    # nckd-like loss
    term1, term2 = method.split('_')
    
    if term1 == "ge":
        mask = _get_ge_mask(logits_teacher, target)
    elif term1 == "both":
        mask = _get_both_mask(logits_teacher, target)
    elif term1 == "g":
        mask = _get_g_mask(logits_teacher, target)
    else:
        raise NotImplementedError
    
    if term2 == "incl":
        pass
    #elif term2 == "excl":
    #    mask = mask.long()
    #    index = torch.argmax(logits_teacher, dim=1, keepdim=True)
    #    mask = torch.where(index == target.unsqueeze(1), torch.zeros_like(mask), mask).bool()
    else:
        raise NotImplementedError

    if loss_type == 'kl':
        assert mask.shape == logits_student.shape
        masked_student = (logits_student / temperature).masked_fill(mask, -1e9)
        log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
        # log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * mask, dim=1)
        masked_teacher = (logits_teacher / temperature).masked_fill(mask, -1e9)
        pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
        # pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * mask, dim=1)
        nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)
    elif loss_type == 'mse':
        masked_student = torch.where(mask, torch.zeros_like(logits_student), logits_student)
        masked_teacher = torch.where(mask, torch.zeros_like(logits_teacher), logits_teacher)
        nckd_loss = F.mse_loss(masked_student, masked_teacher)

    return alpha * tckd_loss + beta * nckd_loss


def _get_both_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    max_index = torch.argmax(logits, dim=1, keepdim=True)
    max_mask = torch.zeros_like(logits).scatter_(1, max_index, 1).bool()
    return gt_mask + max_mask

def _get_ge_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits >= gt_value, 1, 0).bool()
    return mask

def _get_g_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits > gt_value, 1, 0).bool()
    return mask


def sim_loss(logits_student_in, logits_teacher_in, gamma):
    logits_student = F.normalize(logits_student_in, p=2, dim=1)
    logits_teacher = F.normalize(logits_teacher_in, p=2, dim=1)

    sim_student = torch.mm(logits_student.t(), logits_student)
    sim_teacher = torch.mm(logits_teacher.t(), logits_teacher)
    sim_loss = F.mse_loss(sim_student, sim_teacher)
    return gamma * sim_loss


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


class MY(Distiller):

    def __init__(self, student, teacher, cfg):
        super(MY, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MY.CE_WEIGHT
        self.alpha = cfg.MY.ALPHA
        self.beta = cfg.MY.BETA
        #self.gamma = cfg.MY.GAMMA
        self.temperature = cfg.MY.T
        self.warmup = cfg.MY.WARMUP
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.method = cfg.MY.METHOD
        self.loss_type = cfg.MY.LOSS_TYPE
        self.alpha_temperature = cfg.MY.ALPHA_T

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_my = min(kwargs["epoch"] / self.warmup, 1.0) * my_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.logit_stand,
            self.method,
            self.loss_type,
            self.alpha_temperature,
        )
        #loss_sim = min(kwargs["epoch"] / self.warmup, 1.0) * sim_loss(
        #    logits_student,
        #    logits_teacher,
        #    self.gamma,
        #)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_my": loss_my,
            #"loss_sim": loss_sim,
        }
        return logits_student, losses_dict
