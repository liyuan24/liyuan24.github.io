---
layout: post 
title: Supervised Fine Tuning From Scratch
date: 2025-04-09
excerpt: Let's implement SFT from scratch...
---

# Large Language Model(LLM) Training Pipeline

The training pipeline of LLM like chatGPT is a multi-stage process and can be roughly divided into 3 stages:

1. **Pre-training**: next token prediction training on web-scale corpus.
2. **Supervised Fine-tuning(SFT)**: supervised fine-tuning the pre-trained model on conversation dataset. This is to make the model gain the ability to follow the instruction.
3. **Reinforcement Learning with Human Feedback(RLHF)**: Scale the human preference alignment to the LLM. SFT is learning from annotated conversation dataset which is a small dataset and expensive to obtain.


For pre-training, I highly recommend the [GPT2 from scratch tutorial](https://www.youtube.com/watch?v=l8pRSuU81PU) by Andrej Karpathy. In this blog, I will focus on the SFT stage. We will implement it from scratch

1. No use of existing training library, e.g. [Hugging Face trl](https://huggingface.co/docs/trl/en/index).
2. Only ~500 lines of Pytorch code, that is it!

Without further ado, let's get started!

# Is SFT Necessary?

Before we start the implementation, I want to first talk about whether SFT is necessary based on two recent papers.

In DeepSeek R1 paper[1]

> we explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure reinforcement learning process.

The DeepSeek R1-Zero does not use SFT and only use Reinforcement Learning to let the model gain the reasoning ability.

In SFT Memorizes, RL Generalizes[2]

> SFT stabilizes the model’s output format, enabling subsequent RL to achieve its performance gains.

They show that SFT is necessary for the LLM training and will benefit the RL stage.

Are those two papers contradict with each other? Probably not, as the authors of [2] mentioned

> Note that due to the difference in backbone model, our results do not contradict with DeepSeekAI et al. (2025), which suggests that SFT is unnecessary for downstream RL training.

# What is difference between Pre-training and SFT?

It sounds like SFT is really similar to the pre-training. Actually it really is. But with two key differences:

1. **Chat Template**: While the pre-training is predicting the next token, in SFT stage, the model needs to differentiate between who said what. So the training sample need to be structured and have the delimiter of who is the speaker.
2. **Masked Loss Function**: SFT will only calculate the loss on the assistant response tokens without considering the user prompt. This will let the model learn whether the assistant response is good or not.

And that is it!

# Chat Template

As mentioned above, the chat template is letting the model know who said what. A very popular template is from OpenAI and you can find it in the [huggingface trl library](https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L64).

```
 <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you!<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
```

`<im_start>` and `<im_end>` are special tokens and mark the beginning and end of the conversation for each role respectively. 

`user` and `assistant` are the roles of the conversation. The messsage follows `user` is prompt. And the message follows `assistant` LLM response.

Note that different LLM can have different template, e.g. [Llama3](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/). But the purpose is the same.

# Let's implement it from scratch

The source code is available at my github [deepseek_from_scratch](https://github.com/liyuan24/deepseek_from_scratch). Let's talk about each part in the following sections.

## Training Data

We will use the [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset from Hugging Face.

One example is like

```
[ { "content": "Hello, how are you?", "role": "user" }, 
  { "content": "I'm fine, thank you!", "role": "assistant" },
  { "content": "What is the capital of France?", "role": "user" },
  { "content": "The capital of France is Paris.", "role": "assistant" }
]
```

Each item in the list is from one role. The whole list is one conversation.



## Chat Template

The chat template is used to format the training data sample into one string. We use the OpenAI chatML template as mentioned above. So the training sample above will be formated as 

```
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you!<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

## Tokenizer

The tokenizer will be used to convert the string into integers aka tokens. In our tutorial, we will use the GPT4 tokenizer from OpenAI. The source code is at [tokenizer.py](https://github.com/liyuan24/deepseek_from_scratch/blob/main/tokenizer.py). 

One thing to note is that `<|im_start|>` and `<|im_end|>` are not included in the vocabulary of the tokenizer. So we need to add them as special tokens which can be found [here](https://github.com/liyuan24/deepseek_from_scratch/blob/main/tokenizer.py#L24)

## DataCollator

Since different samples in one batch can have different length measured by number of tokens, we need to pad the shorter ones to the max length in this batch. This is done by the DataCollator and can be found [here](https://github.com/liyuan24/deepseek_from_scratch/blob/main/datacollator.py#L63)

And as we mentioned above that the SFT will only calculate the loss on the assistant response tokens without considering the user prompt. So we need to mask the user prompt tokens when calculating the loss. This is also done in the DataCollator. Since we will use [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) of Pytorch, and from the documentation we know that if the label value is `-100`, the loss on that token will be ignored. So in the DataCollator, we will mark the label value from the user prompt as `-100` which can be found [here](https://github.com/liyuan24/deepseek_from_scratch/blob/main/datacollator.py#L120)

## Model

The model is a mini-version of DeepSeek V2[3]. I implemented it from scratch to make sure I understand the model architecture. For official implementation, you can refer to the [code](http://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py) on the Hugging Face.

## Trainer

The [trainer](https://github.com/liyuan24/deepseek_from_scratch/blob/main/trainer.py) is used to train the model with the training data. It is very standard so I won't go into details.


# Future Extension

1. I implemented the full fine-tuning. But some parameter efficient fine-tuning methods are worth to try, like LoRA[4]

2. Distributed training is also a good extension, e.g, [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)







# Reference

\[1\]: @misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}

\[2\]: @misc{chu2025sftmemorizesrlgeneralizes,
      title={SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training}, 
      author={Tianzhe Chu and Yuexiang Zhai and Jihan Yang and Shengbang Tong and Saining Xie and Dale Schuurmans and Quoc V. Le and Sergey Levine and Yi Ma},
      year={2025},
      eprint={2501.17161},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.17161}, 
}

\[3\]: @misc{deepseekai2024deepseekv2strongeconomicalefficient,
      title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2405.04434},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.04434}, 
}

\[4\]: @misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685}, 
}

