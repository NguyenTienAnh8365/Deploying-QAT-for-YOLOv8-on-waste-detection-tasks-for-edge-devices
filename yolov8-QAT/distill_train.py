import argparse
from ultralytics.distill.yolo.distillation_trainer import DistillationTrainer
from ultralytics import settings

from ultralytics.nn.tasks import attempt_load_one_weight, yaml_model_load, guess_model_task
from ultralytics.utils import  DEFAULT_CFG_DICT

def _reset_ckpt_args(args):
    """Reset arguments when loading a PyTorch model."""
    include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
    return {k: v for k, v in args.items() if k in include}

def train(args):
    # Return a specific setting
    settings.update({'datasets_dir': args.dataset_dir})
    # Load a model
    if args.student_weight:
        student_weight, ckpt = attempt_load_one_weight(args.student_weight)
        overrides = student_weight.args = _reset_ckpt_args( student_weight.args)
    else:
        overrides = {}
        overrides['model'] = args.student_config
        overrides['task'] = "detect"
        student_weight=None

    teacher_weight , ckpt = attempt_load_one_weight( args.teacher_weight)

    overrides['data'] = args.dataset_config
    trainer = DistillationTrainer(cfg = DEFAULT_CFG_DICT.copy(), overrides=overrides)
    trainer.model = trainer.get_model(cfg = args.student_config, 
                                      weights = student_weight, 
                                      teacher_cfg=args.teacher_config,
                                      teacher_weight = teacher_weight, 
                                      )
    results = trainer.train()
    return results
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default = '')
    parser.add_argument("--dataset-config", type=str, default = 'coco.yaml')
    parser.add_argument("--teacher-config", type=str, default = 'ultralytics/cfg/models/v8/yolov8s.yaml')
    parser.add_argument("--teacher-weight", type=str, default = 'yolov8s.pt')
    parser.add_argument("--student-config", type=str, default = 'ultralytics/cfg/models/v8/yolov8n.yaml')
    parser.add_argument("--student-weight", type=str, default = '')
    args = parser.parse_args()
    train(args)

