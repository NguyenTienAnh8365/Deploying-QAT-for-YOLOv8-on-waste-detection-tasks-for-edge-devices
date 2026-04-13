import argparse

from ultralytics.qat.pytorch_native.qat_pytorch_trainer import PytorchQuantizationTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import  DEFAULT_CFG_DICT

def _reset_ckpt_args(args):
    """Reset arguments when loading a PyTorch model."""
    include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
    return {k: v for k, v in args.items() if k in include}

def train(args):
    # Return a specific setting
    # Load a mode
    model, ckpt = attempt_load_one_weight('yolov8n.pt')
    task = model.args['task']
    overrides = model.args = _reset_ckpt_args(model.args)
    overrides['model'] = args.model_config
    overrides['data'] = args.data_config
    trainer = PytorchQuantizationTrainer(cfg = DEFAULT_CFG_DICT.copy(), overrides=overrides)
    trainer.model = trainer.get_model(args.model_config,
                                       args.pretrained_weight,
                                       backend = args.quant_backend)

    results = trainer.train()
    return results
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default = 'yolov8s.yaml')
    parser.add_argument("--pretrained-weight", type=str, default = 'yolo8s.pt')
    parser.add_argument("--data-config", type=str, default = 'coco.yaml')
    parser.add_argument("--quant-backend", type=str, default = 'qnnpack')
    args = parser.parse_args()
    train(args)

