---
layout: post 
title: 3 questions about QLoRA
date: 2025-02-17
excerpt: QLoRA is a memory efficient and parameter efficient fine-tuning approach...
---

# Problem Statement

Can we fine-tune a very large model when we have limited GPU memory? For example, for a 70B model, the `float16` weights take 140GB memory, 70B * 2 bytes = 140GB. It is too large to even load into a very advanced GPU with 80GB memory. QLoRA[1] is proposed to solve this problem. Q is *quantization* and LoRA[2] is *low-rank adaptation*. The Q part is to compress the model weights so that a big model can be loaded into limited GPU memory. For example, if we use 4-bit quantization, the 70B model will only take 70B * 0.5 bytes = 35GB memory. The LoRA part is that it doesn't need to fine-tune all the weights of the model, only a small number of additional weights while achieving good performance.

I highly recommend reading the blog from Manal El Aidouni[3] first to understand the details of QLoRA. For quantization, I talked about it in my 8-bit Optimizer blog[4].

And in this blog, I am not going to talk about the details of QLoRA, as Manal has done a great job explaining it in her blog. Instead, I want to talk about my 3 questions when I read the QLoRA paper[1] and Manal's blog[3]. 

1. How is a big model loaded into a single commodity machine?
2. How does the forward pass look like for QLoRA?
3. How does the backward pass look like for QLoRA?

# Let's take a look at concrete examples

## 4-bit quantization

Before talking about my questions, let's first see how 4-bit quantization works. The example is inspired by Manal's blog[3]. I will use the following code to illustrate the idea. Since it is 4-bit, there are 16 quantization candidates. In this case, it is `nf4_quant_levels`. `nf4_quant_index` is just from 0 to 15. `W` is the tensor that we want to quantize. And finally since Pytorch didn't support 4-bit tensor when QLoRA was out, the authors used 8-bit int to store the quantized tensor by packing 2 4-bit values together.

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

To make sure I understand correctly, let's use the official `bitsandbytes` library to do the same thing.

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

## What the model looks like after Quantization

```
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig
from peft import get_peft_model

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
   bnb_4bit_use_double_quant=True)


base_NF4_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b", quantization_config=nf4_config)

base_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")
```

```
print(base_model)

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 3200, padding_idx=0)
    (layers): ModuleList(
      (0-25): 26 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (k_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (v_proj): Linear(in_features=3200, out_features=3200, bias=False)
          (o_proj): Linear(in_features=3200, out_features=3200, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=3200, out_features=8640, bias=False)
          (up_proj): Linear(in_features=3200, out_features=8640, bias=False)
          (down_proj): Linear(in_features=8640, out_features=3200, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((3200,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((3200,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((3200,), eps=1e-06)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3200, out_features=32000, bias=False)
)
```

```
print(base_NF4_model)

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 3200, padding_idx=0)
    (layers): ModuleList(
      (0-25): 26 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=3200, out_features=3200, bias=False)
          (k_proj): Linear4bit(in_features=3200, out_features=3200, bias=False)
          (v_proj): Linear4bit(in_features=3200, out_features=3200, bias=False)
          (o_proj): Linear4bit(in_features=3200, out_features=3200, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=3200, out_features=8640, bias=False)
          (up_proj): Linear4bit(in_features=3200, out_features=8640, bias=False)
          (down_proj): Linear4bit(in_features=8640, out_features=3200, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((3200,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((3200,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((3200,), eps=1e-06)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3200, out_features=32000, bias=False)
)
```

The difference is that for NF4 quantized model, the linear layers in attention layer and MLP are all bitsandbytes `Linear4bit`(4 bit) instead of Pytorch `Linear`. For other modules, like embedding, norm, rotary embedding, etc, they are still `float32`(32 bit).

# 3 questions

In this part, I am going to answer the 3 questions I mentioned in the beginning. I will use the [transformers](https://huggingface.co/docs/transformers/en/index) and [accelerate](https://huggingface.co/docs/accelerate/en/index) lbiraries from HuggingFace to demonstrate.

## Question 1: How is a big model loaded into a single commodity machine?

I am really curious about this question. I just think it is really hard to achieve this. For example, before you loading the weights of the 70B model, you need to first initialize the model. With the naive initialization, this will blow up the memory. So with this question, I started looking at the *transformers* code.

### Use [meta device](https://pytorch.org/docs/stable/meta.html) to create the model

Basically we could choose to create the model without allocating memory to its parameters. In Pytorch, the device the model is created is called meta device. Only the metadata of the model is allocated.

```python
with torch.device('meta'):
    t = torch.tensor([1, 2, 3])
    print(t)
```
Result:
```
tensor(..., device='meta', size=(3,), dtype=torch.int64)
```

Only the `device`, `size` and `dtype` are specificed. And no elements are in this tensor.

In `from_pretrained` function, the model will be created under the [init_empty_weights](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4185) context manager and the model will be created on the meta device.

### Update the related layers with quantized versions
After the model is created, the quantizer will replace the related layers with quantized versions, see [here](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4199)

### Load the model weights with memory mapping
Pytorch [load](https://pytorch.org/docs/stable/generated/torch.load.html) has a flag for whether to use memory mapping to load the weights. With memory mapping, the weights will not move to the CPU memory until you actually use them. In [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L554), it will set this flag to `True` when loading the model.

### Get the memory capacity
There is also a step to check the memory capacity for each available device(CPU, GPU, disk). Those stats will be used to calculate where to place different parts of the model, see [here](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4244).

### Calculate the device map
For each module of the model, the memory needed could be calculated by $num_of_elements * data_type_size$, see [here](https://github.com/huggingface/accelerate/blob/v1.3.0/src/accelerate/utils/modeling.py#L675). So with the memory stats, we could calculate the device map for each module, indicating which device to place the module. Really cool!

And note that in Pytorch, when weights are put on different devices, we need to manually move the input tensors to the correct device in the forward pass. But in backward pass, Pytorch will move the gradients to the correct device automatically. An example is shown below.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 20).to('cpu')
        self.layer2 = nn.Linear(20, 5).to('cuda')

    def forward(self, x):
        x = x.to('cpu')
        x = self.layer1(x)
        x = x.to('cuda')
        x = self.layer2(x)
        return x

# Create the model
model = MyModel()

# Example input data and target
input_data = torch.randn(1, 10, requires_grad=True)  # Batch size of 1, 10 features
target = torch.randn(1, 5)  # Batch size of 1, 5 output features
target = target.to('cuda')

# Forward propagation
output = model(input_data)  # Output is on device2

# Compute loss (on device2)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# Backward propagation
loss.backward()
```

### Load the real weights
Until now, we are sure the memory will not be blown up. Now we could load the pretrained weights. For example the quantized weights will be created [here](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L898) which will reduce the memory usage a lot.

## Question 2: How does the forward pass look like for QLoRA?

Since the quantized weights are not real weights. They are actually the indices of the quantization candidates or quantization levels. In forward pass, those weights should be first be dequantized. If we look at the [4bit matrix multiplication function](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L462) in bitsandbytes code, the guess is verified.

## Question 3: How does the backward pass look like for QLoRA?
Similar to the forward pass, in the backward pass, we need to dequantized weights to calculate the gradients with respect to the input tensors. In Pytorch, this is actually really simply to implement. Just need to create a subclass of `torch.autograd.Function` and override the `forward` and `backward` functions. Check the [backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L476) function in bitsandbytes library. 

# References

\[1\]: @misc{dettmers2023qloraefficientfinetuningquantized,
      title={QLoRA: Efficient Finetuning of Quantized LLMs}, 
      author={Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
      year={2023},
      eprint={2305.14314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.14314}, 
}

\[2\]: @misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685}, 
}

\[3\]: [https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)

\[4\]: [https://liyuan24.github.io/writings/8_bit_optimizer.html](https://liyuan24.github.io/writings/8_bit_optimizer.html)