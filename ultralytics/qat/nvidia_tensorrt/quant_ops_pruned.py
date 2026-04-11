"""
Quantization operations cho pruned YOLOv8 modules.

Adapted từ yolov8-QAT/ultralytics/qat/nvidia_tensorrt/quant_ops.py
cho các module pruned: C2fPruned, BottleneckPruned, SPPFPruned.

Khác biệt chính so với quant_ops.py gốc:
  - C2fPruned dùng .split(cv1_split_sections, dim=1) thay vì .chunk(2, 1)
    vì sau pruning, kênh có thể không đối xứng (vd: [80, 40] thay vì [128, 128])
  - BottleneckPruned có cùng logic residual add
  - Concat và Upsample không thay đổi (không bị pruning)
"""

import torch
import torch.nn.functional as F
from pytorch_quantization.nn.modules import _utils
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn


# =============================================================================
# Quantized forward methods cho pruned modules
# =============================================================================

def bottleneck_pruned_quant_forward(self, x):
    """Quantized forward cho BottleneckPruned — thêm QuantAdd cho residual."""
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def c2f_pruned_quant_forward(self, x):
    """Quantized forward cho C2fPruned — dùng split với sections không đối xứng."""
    if hasattr(self, "c2fchunkop"):
        y = list(self.c2fchunkop(self.cv1(x)))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    else:
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def concat_quant_forward(self, x):
    """Quantized forward cho Concat."""
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)


def upsample_quant_forward(self, x):
    """Quantized forward cho Upsample."""
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x, self.size, self.scale_factor, self.mode)


# =============================================================================
# Quantization wrapper modules
# =============================================================================

class QuantAdd(torch.nn.Module, _utils.QuantMixin):
    """Quantized addition cho residual connection trong BottleneckPruned."""
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantC2fSplitPruned(torch.nn.Module):
    """Quantized split cho C2fPruned — hỗ trợ split sections không đối xứng.

    Khác với QuantC2fChunk gốc (dùng chunk đều 2 phần),
    module này dùng torch.split với sections tùy ý (vd: [80, 40])
    vì sau pruning kênh không còn đối xứng.
    """
    def __init__(self, split_sections):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.split_sections = split_sections

    def forward(self, x):
        return torch.split(self._input0_quantizer(x), self.split_sections, dim=1)


class QuantConcat(torch.nn.Module):
    """Quantized concatenation cho FPN neck."""
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim)


class QuantUpsample(torch.nn.Module):
    """Quantized upsample cho FPN neck."""
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())

    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)


# =============================================================================
# Apply quantization ops to pruned model
# =============================================================================

def quant_module_change_pruned(model):
    """Thêm quantization modules vào pruned model.

    Quét tất cả modules và thay thế forward method bằng phiên bản quantized:
      - C2fPruned  → thêm QuantC2fSplitPruned (quantize trước split)
      - BottleneckPruned → thêm QuantAdd (quantize 2 nhánh trước khi cộng)
      - Concat    → thêm QuantConcat (quantize 2 input trước khi concat)
      - Upsample  → thêm QuantUpsample (quantize trước upsample)

    Lưu ý: Conv2d đã được thay bằng QuantConv2d thông qua quant_modules.initialize()
    trước khi gọi hàm này, nên không cần xử lý Conv ở đây.
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ == "C2fPruned":
            if not hasattr(module, "c2fchunkop"):
                print(f"Add QuantC2fSplitPruned to {name}")
                module.c2fchunkop = QuantC2fSplitPruned(module.cv1_split_sections)
            module.__class__.forward = c2f_pruned_quant_forward

        if module.__class__.__name__ == "BottleneckPruned":
            if module.add:
                if not hasattr(module, "addop"):
                    print(f"Add QuantAdd to {name}")
                    module.addop = QuantAdd(module.add)
                module.__class__.forward = bottleneck_pruned_quant_forward

        if module.__class__.__name__ == "Concat":
            if not hasattr(module, "concatop"):
                print(f"Add QuantConcat to {name}")
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = concat_quant_forward

        if module.__class__.__name__ == "Upsample":
            if not hasattr(module, "upsampleop"):
                print(f"Add QuantUpsample to {name}")
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = upsample_quant_forward
