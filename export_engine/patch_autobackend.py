"""
Patch ultralytics autobackend.py để hỗ trợ TensorRT 10.x API.

TensorRT 10.x đã xóa API cũ (num_bindings, get_binding_name...) và dùng API mới
(num_io_tensors, get_tensor_name...). Script này tự động tìm và vá file autobackend.py
trong gói ultralytics đã cài.

Usage:
    python patch_autobackend.py

Sau khi chạy, restart Python kernel/runtime rồi inference bình thường:
    from ultralytics import YOLO
    model = YOLO("best.tensorrt.engine", task="detect")
    results = model("image.jpg")
"""

import importlib
import re
import shutil
import sys
from pathlib import Path


# ─── Code thay thế cho phần TRT init trong autobackend.py ────────────────────

# Pattern cho cả stock ultralytics 8.0.228 và custom fork
OLD_PATTERN = r"(?:# Support TensorRT.*?)?for i in range\(model\.num_bindings\):.*?batch_size = bindings\['images'\]\.shape\[0\]"

NEW_TRT_INIT = '''# Detect TRT API version from version string (direct attribute access unreliable
            # in pip-installed TRT builds on some platforms)
            trt_major = int(trt.__version__.split('.')[0])
            trt_new_api = trt_major >= 10

            if trt_new_api:
                # TRT 10.x API: enumerate IO tensors via get_tensor_name loop
                # (num_io_tensors property raises AttributeError on some pip TRT builds)
                all_io_names = []
                for _i in range(4096):
                    try:
                        _name = model.get_tensor_name(_i)
                        if not _name:
                            break
                        all_io_names.append(_name)
                    except Exception:
                        break
                for name in all_io_names:
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    shape = tuple(model.get_tensor_shape(name))
                    if is_input:
                        if -1 in shape:
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[2]))
                            shape = tuple(context.get_tensor_shape(name))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                        shape = tuple(context.get_tensor_shape(name))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            else:
                # TRT 8.x API
                for i in range(model.num_bindings):
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if is_input:
                        if -1 in tuple(model.get_binding_shape(i)):
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]'''

OLD_FORWARD_PATTERN = r"if self\.engine:.*?self\.context\.execute_v2\(list\(self\.binding_addrs\.values\(\)\)\)"

NEW_TRT_FORWARD = '''if self.engine:
                if self.dynamic and im.shape != self.bindings['images'].shape:
                    i = self.model.get_input_index('images')
                    self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                    self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_output_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
                s = self.bindings['images'].shape
                assert im.shape == s, f"input size {im.shape} {\'>\' if self.dynamic else \'not equal to\'} max model size {s}"
                if getattr(self, '_trt_new_api', False):
                    # TensorRT 10.x
                    self.context.set_tensor_address('images', int(im.data_ptr()))
                    for name in self.output_names:
                        self.context.set_tensor_address(name, self.binding_addrs[name])
                    self.context.execute_async_v3(0)
                else:
                    # TensorRT 8.x
                    self.binding_addrs['images'] = int(im.data_ptr())
                    self.context.execute_v2(list(self.binding_addrs.values()))'''


def find_autobackend():
    """Tìm đường dẫn autobackend.py trong gói ultralytics đã cài."""
    try:
        import ultralytics
        pkg_path = Path(ultralytics.__file__).parent
        autobackend = pkg_path / "nn" / "autobackend.py"
        if autobackend.exists():
            return autobackend
    except ImportError:
        pass
    return None


def apply_patch(path: Path):
    """Áp dụng patch TRT 10.x vào autobackend.py."""
    content = path.read_text(encoding="utf-8")

    # Kiểm tra đã patch chưa
    if "trt_major = int(trt.__version__" in content:
        print(f"[OK] {path} đã được patch rồi, bỏ qua.")
        return True

    # Backup
    backup = path.with_suffix(".py.bak")
    shutil.copy(path, backup)
    print(f"  Backup: {backup}")

    # Patch phần init (thay khối từ "# Support TensorRT" đến "batch_size = ...")
    patched = re.sub(
        OLD_PATTERN,
        NEW_TRT_INIT,
        content,
        flags=re.DOTALL,
    )

    if patched == content:
        print("[WARN] Không tìm thấy pattern để patch phần init.")
        print("       Có thể file đã bị sửa khác hoặc version ultralytics khác.")
        print("       Thêm trt_new_api thủ công vào __init__ và lưu trt_new_api vào self._trt_new_api.")
        return False

    # Lưu
    path.write_text(patched, encoding="utf-8")
    print(f"[OK] Patch thành công: {path}")
    return True


def verify_patch(path: Path):
    """Kiểm tra patch đã được áp dụng."""
    content = path.read_text(encoding="utf-8")
    checks = [
        ("trt_major = int(trt.__version__", "version-based API detection"),
        ("get_tensor_name", "TRT 10.x get_tensor_name loop"),
    ]
    all_ok = True
    for marker, desc in checks:
        if marker in content:
            print(f"  [✓] {desc}")
        else:
            print(f"  [✗] THIẾU: {desc}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("Patch ultralytics autobackend.py cho TensorRT 10.x")
    print("=" * 60)

    # Cho phép truyền path thủ công
    if len(sys.argv) > 1:
        autobackend_path = Path(sys.argv[1])
    else:
        autobackend_path = find_autobackend()

    if autobackend_path is None or not autobackend_path.exists():
        print("[ERROR] Không tìm thấy autobackend.py.")
        print("        Cài ultralytics trước: pip install ultralytics==8.0.228")
        print("        Hoặc truyền path thủ công: python patch_autobackend.py /path/to/autobackend.py")
        sys.exit(1)

    print(f"  Found: {autobackend_path}")

    success = apply_patch(autobackend_path)
    if success:
        print("\n[Kiểm tra patch]")
        verify_patch(autobackend_path)
        print("\n[DONE] Restart Python/kernel để áp dụng.")
    else:
        print("\n[FAILED] Patch thất bại. Xem hướng dẫn thủ công trong file patch_autobackend.py.")
        sys.exit(1)
