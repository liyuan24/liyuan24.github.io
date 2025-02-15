---
layout: post 
title: Let's deep dive into QLoRA
date: 2025-02-16
excerpt: QLoRA is a method to quantize the weights of the model...
---

# QLoRA

QLoRA is a method to quantize the weights of the model. It is a combination of quantization and LoRA.

# Experiment

```python
import torch
nf4_quant_levels = torch.tensor(
    [-1.0, -0.6961928009986877, -0.5250730514526367,
     -0.39491748809814453, -0.28444138169288635, 
     -0.18477343022823334, -0.09105003625154495, 
     0.0, 0.07958029955625534, 0.16093020141124725, 
     0.24611230194568634, 0.33791524171829224, 
     0.44070982933044434, 0.5626170039176941, 
     0.7229568362236023, 1.0]
    )
nf4_quant_index = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
    dtype=torch.uint8)
W = torch.randn(5, 4)
flat_W = W.flatten()
max_val = flat_W.abs().max()
normalized_W = flat_W / max_val
quantized_4bit_W = torch.zeros_like(flat_W, dtype=torch.uint8)
for i, val in enumerate(normalized_W):
    quantized_4bit_W[i] = torch.argmin(torch.abs(nf4_quant_levels - val))
packed_W_8bits = []
for i in range(0, quantized_4bit_W.shape[0], 2):
    packed_W_8bits.append((quantized_4bit_W[i] << 4) | quantized_4bit_W[i+1])
packed_W_8bits = torch.tensor(packed_W_8bits, dtype=torch.uint8)
packed_W_8bits
```
Result:
```
tensor([214, 239,  18, 215, 102, 246, 154, 133,  86,  56], dtype=torch.uint8)
```

```python
import bitsandbytes as bnb
bnb_parameter_4bit = bnb.nn.Params4bit(W, quant_type='nf4').to('cuda')
bnb_parameter_4bit.data.flatten()
```
Result:
```
tensor([214, 239,  18, 215, 102, 246, 154, 133,  86,  56], device='cuda:0',
       dtype=torch.uint8)
```

