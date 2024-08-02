import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

verify_dict = {
    "res32x4": "resnet32x4",
    "res8x4": "resnet8x4",
    "shuv2": "ShuffleV2",
    "res18": "ResNet18",
    "res34": "ResNet34",
    "res50": "ResNet50",
    "mv1": "MobileNetV1",
    "mv2": "MobileNetV2",
    "res56": "resnet56",
    "res20": "resnet20",
    "res110": "resnet110",
    "res32": "resnet32",
    "vgg13": "vgg13",
    "vgg8": "vgg8",
    "wrn_40_2": "wrn_40_2",
    "wrn_16_2": "wrn_16_2",
    "wrn_40_1": "wrn_40_1",
    "dkd": "DKD",
    "kd": "KD",
    "rc": "RC",
    "mlkd": "MLKD",
    "mlkd_noaug": "MLKD_NOAUG",
    "my": "MY",
    "vanilla": "NONE",
    "swap": "SWAP",
    "revision": "REVISION",
}

def main(cfg, resume, opts):
    tags = cfg.EXPERIMENT.TAG.split(",")

    type_name = tags[0]
    teacher_name = tags[1]
    student_name = tags[2]
    assert verify_dict[type_name] == cfg.DISTILLER.TYPE
    assert verify_dict[teacher_name] == cfg.DISTILLER.TEACHER
    assert verify_dict[student_name] == cfg.DISTILLER.STUDENT
    if cfg.DATASET.TYPE == "cifar100":
        if student_name in ["shuv1", "shuv2", "mv2"]:
            assert cfg.SOLVER.LR == 0.01
        else:
            assert cfg.SOLVER.LR == 0.05

    experiment_name_for_output = os.path.join(cfg.EXPERIMENT.PROJECT, ",".join([teacher_name, student_name]), type_name)
    if ",".join(tags[3:]) == "":
        experiment_name_for_output = os.path.join(experiment_name_for_output, "base")
    else:
        experiment_name_for_output = os.path.join(experiment_name_for_output, ",".join(tags[3:]))

    ''' # disable wandb
    if cfg.LOG.WANDB:
        experiment_name = cfg.EXPERIMENT.NAME
        if experiment_name == "":
            experiment_name = cfg.EXPERIMENT.TAG
        if opts:
            addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
            tags += addtional_tags
            experiment_name += ",".join(addtional_tags)
        experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
        try:
            import wandb
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False
    '''
    
    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name_for_output, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)

# for cifar100
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--same-t", action="store_true")
    parser.add_argument("--base-temp", type=float, default=2.)
    parser.add_argument("--kd-weight", type=float, default=9.)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.aug:
        cfg.EXPERIMENT.AUG = True
        cfg.EXPERIMENT.TAG += ',aug'

    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD', 'DKD', 'MLKD', 'MY', 'SWAP', 'RC', 'MLKD_NOAUG']:
        cfg.EXPERIMENT.LOGIT_STAND = True
        cfg.EXPERIMENT.TAG += ',stand'
        if cfg.DISTILLER.TYPE == 'KD' or cfg.DISTILLER.TYPE == 'SWAP' or cfg.DISTILLER.TYPE == 'MLKD' or cfg.DISTILLER.TYPE == 'MLKD_NOAUG':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'MY':
            cfg.MY.ALPHA = cfg.MY.ALPHA * args.kd_weight
            cfg.MY.BETA = cfg.MY.BETA * args.kd_weight
            cfg.MY.T = args.base_temp
            cfg.MY.ALPHA_T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'RC':
            cfg.RC.KD_WEIGHT = cfg.RC.KD_WEIGHT * args.kd_weight
            cfg.RC.RC_WEIGHT = cfg.RC.RC_WEIGHT * args.kd_weight
            cfg.RC.T = args.base_temp
    
    if cfg.DISTILLER.TYPE in ['MY']:
        assert cfg.MY.METHOD in ['both_excl', 'both_incl', 'ge_excl', 'ge_incl', 'g_incl']
        assert cfg.MY.LOSS_TYPE in ['kl', 'mse']
        cfg.EXPERIMENT.TAG += ',' + cfg.MY.LOSS_TYPE
        if args.same_t:
            cfg.MY.ALPHA_T = cfg.MY.T
        if cfg.MY.LOSS_TYPE == 'kl':
            cfg.EXPERIMENT.TAG += ',at=' + str(cfg.MY.ALPHA_T)
            cfg.EXPERIMENT.TAG += ',t=' + str(cfg.MY.T)
            cfg.EXPERIMENT.TAG += ',alpha=' + str(cfg.MY.ALPHA)
            cfg.EXPERIMENT.TAG += ',beta=' + str(cfg.MY.BETA)
            #cfg.EXPERIMENT.TAG += ',' + str(cfg.MY.GAMMA)
            cfg.EXPERIMENT.TAG += ',' + cfg.MY.METHOD
        elif cfg.MY.LOSS_TYPE == 'mse':
            cfg.EXPERIMENT.TAG += ',' + str(cfg.MY.ALPHA)
            cfg.EXPERIMENT.TAG += ',' + str(cfg.MY.BETA)
            #cfg.EXPERIMENT.TAG += ',' + str(cfg.MY.GAMMA)
            cfg.EXPERIMENT.TAG += ',' + cfg.MY.METHOD

    if cfg.DISTILLER.TYPE in ['DKD']:
        assert cfg.DKD.LOSS_TYPE in ['kl', 'mse']
        cfg.EXPERIMENT.TAG += ',' + cfg.DKD.LOSS_TYPE
        if cfg.DKD.LOSS_TYPE == 'kl':
            cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.T)
            cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.ALPHA)
            cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.BETA)
        elif cfg.DKD.LOSS_TYPE == 'mse':
            cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.ALPHA)
            cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.BETA)

    cfg.freeze()
    main(cfg, args.resume, args.opts)
