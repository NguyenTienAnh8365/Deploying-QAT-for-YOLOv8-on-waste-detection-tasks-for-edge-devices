from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
import pytorch_quantization
from ultralytics.nn.tasks import DetectionModel
from .quant_ops import quant_module_change
import torch
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from pathlib import Path


def export_onnx_nvidia( cfg=None, weights=None, nc = 80, format = "onnx" ,verbose=True):
    """_summary_

    Args:
        cfg (_type_, optional): Student model configuration. Defaults to None.
        weights (_type_, optional): Student weight configuration. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()
    # Do calibration x
    model = DetectionModel(cfg = cfg, ch = 3, nc= nc, verbose = False)
    quant_module_change(model)
    # Handle both full checkpoint (best.pt) and raw state_dict (_temp_calibration.pt)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model'].float().state_dict()
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    output_path = Path(weights).with_suffix(".tensorrt.onnx")
    
    for m in model.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
            m.export = True
            m.format = format
    dummy_input = torch.randn(1, 3, 640, 640)

    input_names = [ "image" ]
    output_names = [ "output" ]

        # enable_onnx_checker needs to be disabled. See notes below.
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        verbose=False, 
        opset_version=14, 
        input_names=input_names,
        output_names=output_names
        )
    return model