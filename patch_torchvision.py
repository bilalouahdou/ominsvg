#!/usr/bin/env python3
"""Patch basicsr for torchvision 0.17.0 compatibility"""
import os
import torch

# Create stub for functional_tensor module
tp = os.path.join(torch.__path__[0], 'transforms')
os.makedirs(tp, exist_ok=True)

with open(os.path.join(tp, 'functional_tensor.py'), 'w') as f:
    f.write('# Stub for backwards compatibility with basicsr\n')
    f.write('# functional_tensor was removed in torchvision 0.17.0\n')

print('Patched torchvision.transforms.functional_tensor')
