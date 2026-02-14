#!/usr/bin/env python3
"""Patch basicsr for torchvision 0.17.0 compatibility.

In torchvision >= 0.17.0, the module `torchvision.transforms.functional_tensor`
was removed. basicsr still imports `rgb_to_grayscale` from it, so we create a
shim that re-exports the function from its new location.
"""
import os
import torchvision

transforms_dir = os.path.join(os.path.dirname(torchvision.__file__), 'transforms')
stub_path = os.path.join(transforms_dir, 'functional_tensor.py')

stub_code = """\
# Auto-generated shim for basicsr compatibility.
# torchvision.transforms.functional_tensor was removed in torchvision 0.17.0.
# Re-export rgb_to_grayscale from its new home.
from torchvision.transforms.functional import rgb_to_grayscale  # noqa: F401
"""

with open(stub_path, 'w') as f:
    f.write(stub_code)

print(f'Patched torchvision.transforms.functional_tensor at {stub_path}')
