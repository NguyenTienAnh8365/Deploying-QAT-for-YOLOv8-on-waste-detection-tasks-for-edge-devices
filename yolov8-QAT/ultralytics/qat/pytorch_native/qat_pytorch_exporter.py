from .qat_pytorch_trainer import load_quantized_model
import torch
from pathlib import Path 

def export_onnx_native(cfg, weight, nc):
    model = load_quantized_model(weight, cfg, nc)
    model.model[-1].export = "qat"
    dummy_input = torch.randn(1, 3, 640, 640)
    output_path = Path(weight).with_suffix(".onnx")

    model(dummy_input)

    input_names = [ "image" ]
    output_names = [ "output" ]

    torch.onnx.export(
        model, 
        dummy_input,
        output_path, 
        verbose=False, 
        opset_version=17, 
        input_names=input_names,
        output_names=output_names
        ) 