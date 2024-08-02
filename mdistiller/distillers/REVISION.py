import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def revision_loss(logits_student_in, logits_teacher_in, target, lambda1, lambda2, eta, num_classes):
    _, max_index = torch.max(logits_teacher_in, dim=1)
    right_mask = (target == max_index)
    wrong_mask = (target != max_index)
    right_logits_teacher = logits_teacher_in[right_mask]
    wrong_logits_teacher = logits_teacher_in[wrong_mask]
    right_logits_student = logits_student_in[right_mask]
    wrong_logits_student = logits_student_in[wrong_mask]
    right_target = target[right_mask]
    wrong_target = target[wrong_mask]
    if len(right_target) > 0:
        ce_loss = F.cross_entropy(right_logits_student, right_target)
        mse_loss1 = F.mse_loss(right_logits_student, right_logits_teacher)
    else:
        ce_loss = torch.zeros([1], device=target.device)
        mse_loss1 = torch.zeros([1], device=target.device)
    
    if len(wrong_target) > 0:
        wrong_pred_teacher = F.softmax(wrong_logits_teacher, dim=1)
        wrong_pred_student = F.softmax(wrong_logits_student, dim=1)
        one_hot_wrong_target = F.one_hot(wrong_target, num_classes)
        max_pred, _ = torch.max(wrong_pred_teacher, dim=1, keepdim=True)
        gt_pred = torch.gather(wrong_pred_teacher, dim=1, index=wrong_target.unsqueeze(1))
        beta = eta / (max_pred - gt_pred + 1)
        wrong_pred = beta * wrong_pred_teacher + (1 - beta) * one_hot_wrong_target
        mse_loss2 = F.mse_loss(wrong_pred_student, wrong_pred)
    else:
        mse_loss2 = torch.zeros([1], device=target.device)

    return ce_loss + lambda1 * mse_loss1 + lambda2 * mse_loss2


class REVISION(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(REVISION, self).__init__(student, teacher)
        self.lambda1 = 4.
        self.lambda2 = 1.
        self.eta = 0.8
        self.num_classes = 1000 if cfg.DATASET.TYPE == "imagenet" else 100

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        overall_loss = revision_loss(logits_student, logits_teacher, target, self.lambda1, self.lambda2, self.eta, self.num_classes)
        losses_dict = {
            "loss": overall_loss,
        }
        return logits_student, losses_dict
