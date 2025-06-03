---
layout: post 
title: Vision Language Model
date: 2025-06-02
excerpt: A few papers about vision language model
---

# Vision Language Model

Vision language model is a way to let visual information interact with language information. It is not image generation model like given a prompt and generating an image. But it can generate the image caption, answer the question about the image, etc.

# CLIP: Learning Transferable Visual Models From Natural Language Supervision[1]

CLIP is a fundamental work for learning the image representation from the natural language supervision. Traditionally, there are many tasks in the computer vision field, e.g. image classification, image segmentation, etc. For each task, a good model needs a lot of crowd-labeled data to learn. And different tasks need different labelled data. So the author thought if they could train a model that can learn image representation which is task-agnostic, then they could use the same image representation for different tasks. This idea is actually very similar to the pre-training language model like BERT(masked language modeling) and GPT(auto-regressive language modeling).

The objective that a pre-training model learns is usually not very interesting. For example, the objective of BERT is to predict the masked word(token) in the sentence. The objective of GPT is to predict the next word(token) in the sentence. But they are very powerful because they learn a good language representation and the downstream tasks can be fine-tuned with the pre-trained model. Why the pre-training model can learn a good representation? Because they learn from a massive amount of data on the web. And why massive data? Because **no human label is needed**.

CLIP is motivated by the similar idea. There are massive amount of image and its description on the web. So the author thought the image representation can be learned with the supervision of the text description. They collected a 400 million image-text pair dataset.

Notice that there are also self-supervised or unsupervised models on image like MoCo[2] and MAE[3]. CLIP has another advantage is that 

> it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer.

Expanding on the zero-shot transfer with the image classification task as an example. For MoCo and MAE, they can learn a good image representation. To do image classification, either linear probing or fine-tuning is needed with the labelled data. But for CLIP, they can directly use the pre-trained CLIP model to do the image classification. The text input can just be `a photo of a {class_name}`.

## Dataset

They collected a 400 million **image-text pair** dataset from the internet. For reference, the popular training set for image related tasks are 
* MS-COCO: 100K images
* Visual Genome: 100K images
* Full ImageNet Dataset: 14 million images

## Model Architecture
So they first considered predicting the image description. But found that it was really slow. Then they considered the contrastive learning and found that it was much more efficient. With contrastive learning, the model doesn't learn to predict the image description. Instead one image encoder and one text encoder are used to encode the image and text respectively. Then the image and text representations learned from the two encoders are projected to a joint embedding space. The objective is to maximize the cosine similarity between the matched image-text pair and minimize the similarity between the mismatched image-text pair.

### Image Encoder

They tried ResNet[4] and ViT[5]

* Input: Image, tensor of shape `[B, C, H, W]`
    * `B`: batch size
    * `C`: channel
    * `H`: height
    * `W`: width
* Output: Image representation(embedding), tensor of shape `[B, D]`
    * `D`: dimension of the embedding

### Text Encoder

They use the same architecutre as in GPT2[6]. 

1. Attention is causal mask attention. So this is a **decoder-only** transformer.
2. The text representation is the hidden state of the last layer of the transformer of the last token([EOS]).

* Input: Tokenized text, tensor of shape `[B, L]`
    * `B`: batch size
    * `L`: max context length
* Output: Text representation(embedding), tensor of shape `[B, D]`
    * `D`: dimension of the embedding

### Projection to joint embedding space

A linear projection layer is used to project the image and text representations to the joint embedding space.

### Objective

* Loss function: cross-entropy loss

> Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible (image, text) pairings across a batch actually occurred. To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N^2 − N incorrect pairings.

### Learned temperature

A learned temperature is used to scale the cosine similarity. And the temperature is initialized to 0.07.

## Evaluation

They first evaluate CLIP's zero-shot capability on the image classification task.

1. Use label directly as the text description: compute the label representations and the image representation. Then compute the cosine similarity between the label representation and the image representation. Predict the class with the highest cosine similarity.
2. Construct the prompt: `a photo of a {class_name}`. This yield better performance. They also found that for different task, constructing different prompt is better.

Then they also evaluate the linear probing by freezing the image encoder and add a trainable linear layer on top of it.

## Limitation

As mentioned in Flamingo[7], since CLIP only outputs the embeddings of text and image, it can only be used in limited tasks such as image classification where a finite set of outcomes is provided beforehand. This also reminds me that when reading a paper, it would be good to think about the limitation of the work because that is where an improvement can be made.

# Flamingo: a Visual Language Model for Few-Shot Learning[7]

Flamingo is another foundamental paper in the vision language model field. It is a langugage generation model that is conditioned on the visual information. It can solves a larger set of tasks than CLIP such as image captioning and visual question answering. Also note that the visual and text encoders are pre-trained and frozen during the training of the Flamingo model.

It highlights the few-shot capability by giving a few examples, the model can generate the desired output for the user query. This capability is from the large scale training on the text data. 
> This training provides general-purpose generation capabilities that allows these LMs to perform well when prompted with task examples

And they leverage Perceiver[8] to encode the visual information. This is important because regardless of the resolution of the image, the Perceiver can encode the visual information as a fixed-size vector(image tokens). Then those image tokens are cross-attented with the text tokens to mix the multi-modal information.

## Dataset

They training dataset comes from 3 sources:
1. Massive MultiModal Web(M3W): this is the scaping of the web pages and contains the image text interleaved documents. 43 million webpages.
2. LTIP (Long Text & Image Pairs) consists of 312 million images, 
3. VTP (Video & Text Pairs) consists of 27 million short videos (approximately 22 seconds on average).

For M3W, because the image associated with the related text is not deterministic, they use the heuristic to associate the text tokens with the image either preceding or following that token. In training, the probability of the image preceding the text is 0.5. Note that in inference, the image is always preceding the text which is different from the training.

So to summarize the training dataset, it is the image-text interleaved sequence. The input text to the language model is like

```
<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|>
```

The images are replaced with the `<image>` token and they also add a `<|endofchunk|>` token to indicate the end of an image-text chunk.

## Model Architecture

### Visual Processor

Visual processor includes two components:
1. Image Encoder: encode the visual information
2. Perceiver: turn variable size of visual representations into a fixed number of representations to reduce the computation cost of the cross-attention.

#### Image Encoder

This is a ResNet. It is first pre-trained on the image-text pairs dataset using the contrastive learning which is the same as CLIP above. Then it is frozen and not updated during the training of the Flamingo model. For image, it is straight forward. But for video, frames are sampled at 1 fps and encoded separately. 

* Input: a sequence of images, tensor of shape `[B, T, C, H, W]`
    * `B`: batch size
    * `T`: number of frames, for image, `T=1`
    * `C`: channel
    * `H`: height
    * `W`: width
* Output: a sequence of image tokens, tensor of shape `[B, T, S, D]`
    * `S`: number of image tokens
    * `D`: dimension of the image token

#### Perceiver[8]

As mentioned above, the output of the image encoder is of shape `[B, T, S, D]` where `T` and `S` might be very large. To save computation, Perceiver is used to turn variable size of visual representations into a fixed number of representations. I highly recommend reading the Perceiver paper to understand the details.

* Input: a sequence of image tokens, tensor of shape `[B, T, S, D]`
    * `B`: batch size
    * `T`: number of frames
    * `S`: number of image tokens
    * `D`: dimension of the image token
* Output: a shorter sequence of image tokens, tensor of shape `[B, T, num_latents, D]`
    * `num_latents`: number of compressed representations which is much less than `S`

### Text Decoder

The text decoder is a decoder-only transformer. And to mix the information from the visual processor, the image tokens are cross-attented with the text tokens. Specifically, the text tokens are queries and image tokens are keys and values. And text tokens will only attend to the *associated* image tokens. So this is a **masked** cross-attention.

During training, only the cross-attention layer is updated and all other modules are frozen. This is to prevent the *catastrophic forgetting* of the pre-trained LLM.
* Input: a sequence of text tokens, tensor of shape `[B, L]`
    * `B`: batch size
    * `L`: max context length
* Output: a sequence of text tokens, tensor of shape `[B, L, D]`
    * `D`: dimension of the text token

### Objective

The objective of Flamingo is the standard auto-regressive language modeling objective, cross-entropy loss.

## Evaluation

They evaluate Flamingo model on a bunch of visual-language benchmarks with **few-shot** setting.

## How image and text representations are aligned?

Since there is no contrastive learning in Flamingo, I am curious how the image and text representations are aligned. The image representations are cross-attented with the text representations, but those 2 representations are from different encoders and similar concepts are not close to each other in the embedding space. 

Flamingo should be using the image grounded text generation loss to do the alignment. Note that the cross attention layer is updated during the training. And the query, key, value linear projections should be learned to align the image and text representations.

## Limitation

Not good at image text retrieval tasks.

# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation[9]

CLIP generates the image and text representations and Flamingo generates the text based on the image. So how about having a model that can do both? This is the motivation of BLIP. There are 3 tasks in BLIP

1. Image text contrastive learning which is the same as CLIP
2. Image grounded text encoder: this is a binary classifier that outputs whether the text is the caption of the image
3. Image grounded text decoder: this is a text decoder that generates the caption of the image

And from model perspective, BLIP have 3 components:
1. Image encoder
2. Text encoder
3. Text decoder

Another contribution of BLIP is that it is trying to improve the image-text pair data quality by training a captioner to generate the high quality caption of the image and a filter to remove the noisy captions.

## Dataset

There are 2 stages of pre-training.

### Stage 1: noisy image-text pairs

| Dataset | # Images | # Texts |
|---------|----------|---------|
| COCO    | 113K     | 567K    |
| VG      | 100K     | 769K    |
| SBU     | 860K     | 860K    |
| CC3M    | 3M       | 3M      |
| CC12M   | 10M      | 10M     |
| LAION   | 115M     | 115M    |

### Stage 2: data processed by the captioner and filter

In the 2nd stage of pre-training, the trained captioner and filter(more details below) are used to genreate the synthetic caption and filter the noisy captions.


## Model Architecture

### Image Encoder
It is a ViT[5] with a `cls` token for the final image representation.

* Input: Image, tensor of shape `[B, C, H, W]`
    * `B`: batch size
    * `C`: channel
    * `H`: height
    * `W`: width
* Output: Image representation, tensor of shape `[B, D]`
    * `D`: dimension of the image representation

### Text Encoder
It is a BERT[10] model. This BERT model is used for both text encoder in the image-text contrastive learning and the image grounded text encoder but with a few differences:

1. For contrastive learning, a `[CLS]` token is appended to the beginning of the text input to summarize the sentence
2. For image grounded text encoder, a `[ENCODE]` token is added at the beginning of the text input and that token is finally used as the representation. Also to mix the image information, a cross-attention layer is added between the self-attention layer and the mlp layer.

For contrastive learning,
* Input: Tokenized text, tensor of shape `[B, L]`
    * `B`: batch size
    * `L`: max context length
* Output: Text representation, tensor of shape `[B, D]`
    * `D`: dimension of the text representation

For image grounded text encoder, a classification head is added on top of the text encoder.
* Input: Tokenized text, tensor of shape `[B, L]` and image representation, tensor of shape `[B, D]`
    * `B`: batch size
    * `L`: max context length
    * `D`: dimension of the image representation
* Output: Binary classification, tensor of shape `[B, 2]`

### Text Decoder
A standard transformer decoder and use causal mask attention instead of the bi-directional attention in BERT.

* Input: Tokenized text, tensor of shape `[B, L]` and image representation, tensor of shape `[B, D]`
    * `B`: batch size
    * `L`: max context length
    * `D`: dimension of the image representation
* Output: generated text, tensor of shape `[B, l]`
    * `l`: length of the generated text

### Objective

This is a multi-task learning model. Each example is a pair of image and text and it is used for all the 3 tasks.

* For image text contrastive learning, the objective is the same as CLIP, cross-entropy loss.
    * A typical issue in contrastive learning is false negative. More specifically, in one batch of size `N`, there are `N` positve pairs and `N^2-N` negative pairs. But there could be some pairs in the `N^2-N` negative pairs that are actually positive pairs. To mitigate this issue, BLIP is using a soft-label generated by a **Momentum** encoder. The idea is from the paper MoCo[2]. Please refer to the paper for more details.
    * Also the **Momemtum** queue provide a larger number of negative pairs for the contrastive learning. Since BLIP is trained end to end(image and text encoders are not frozen), a large batch size cannot be used which means that in-batch negative pairs may not be enough.
* For image grounded text encoder, the objective is the standard binary classification objective, cross-entropy loss.
  * Instead of using the `N(N-2)` negative pairs in CLIP, BLIP is using **hard negative mining** to select the most difficult negative pairs for each image and text. So there are total `N` positive pairs and `2 * N` negative pairs. The measurement of the difficulty is from the cosine similarity between the image and text representations in contrastive learning results in this batch.
* For image grounded text decoder, the objective is the standard auto-regressive language modeling objective, cross-entropy loss.
  * A label smooth of `0.1` is used to prevent the model from over-confident.

### Parameter Sharing

To improve the pre-training efficiency, the text encoder and text decoder share the parameters except the self-attention layer because encoder is using bi-directional attention and decoder is using causal mask attention.

### Captioner and Filter

As mentioned before, BLIP is using a captioner to generate the high quality caption of the web images and a filter to remove the noisy captions of the web and synthetic captions to improve the pre-training data quality. 

The captioner is inititalized from the text decoder of the pre-trained BLIP on noisy image-text pairs. Filter is initialized from the text encoder of the pre-trained BLIP on noisy image-text pairs.

Both of them are finetuned individually on the COCO dataset.

## Evaluation

1. Image-Text Retrieval
2. Image Captioning
3. Visual Question Answering (VQA)
4. Natural Language Visual Reasoning
5. Visual Dialog

## Limitation

Both the image and text encoders are trained end to end although initalized from the pre-trained models. This is very resource-intensive.

# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models[11]

As we mentioned above both image and text encoders needs to be trained end to end in BLIP which is very resource-intensive. BLIP-2 is trying to solve this problem by using a frozen image encoder and a frozen large language model. But since two encoders are not related, their representations are not aligned. So how to do the alignment? A very simple idea is to just use a MLP layer to project the image representation to the text representation space. BLIP-2 is using a more advance way called **Q-former** to do this.

Q-former is actually a BERT transformer model. And as in Flamingo, a fixed set of **queries** are learned so that regardless of the resolution of the image, only a fixed set of image representations will be learned. 

## Dataset

The training dataset is the same as BLIP which is a large set of image-text pairs.

## Model Architecture

The goal of Q-former is extract the image information that is most relevent to the text. To achieve this, they use the same objectives as in BLIP, a contrastive loss, a image-text matching loss and a text generation loss.

### Q-former: first stage pre-training

Q-formare is a [BERT model](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2.py#L54). In this stage, a fixed set of learnable queries are learned which are used to extract the image information that is most relevent to the text.

#### Objectives

Image is first encoded by the frozen image encoder.

* Contrastive loss
  * Both text and the learned queries are passed through the Q-former. And the image representation is cross-attended with the learned queries.
  * Since there are `N` queries embeddings finally, they first compute the pair-wise similarity between the text representation and the `N` queries embeddings.
  * Then they use the highest similarity as the output
  * The loss is the standard cross-entropy loss
* Image-text matching loss
  * Learnable queries are added to the start of the text embeddings after tokenization. 
  * Image representation is cross-attended with both learned queries and the text embeddings.
  * Binary classification
  * Use hard-negative mining same as BLIP
* Text generation loss
  * The learnable queries output is used as KV cache for the text decoder(Q-former) so that the text generation is conditioned on the image information. See [code](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py#L263)
  * Standard language modeling loss

### Second stage pre-training

In this stage, the Q-former output will be projected to the text representation space of the frozen large language model with a MLP layer.

> The projected query embeddings are then prepended to the input text embeddings. They function as soft visual prompts that condition the LLM on visual representation extracted by the Q-Former. Since the Q-Former has been pre-trained to extract language-informative visual representation, it effectively functions as an information bottleneck that feeds the most useful information to the LLM while removing irrelevant visual information. This reduces the burden of the LLM to learn vision-language alignment, thus mitigating the catastrophic forgetting problem.

* Input: text, tensor of shape `[B, L]` and projected query embeddings, tensor of shape `[B, N, D]`
  * `B`: batch size
  * `L`: max context length
  * `N`: number of learnable queries
  * `D`: dimension of the learnable queries
* Output: generated text, tensor of shape `[B, l]`
  * `l`: length of the generated text

#### Objectives

Standard language modeling loss

## Evaluation

* Zero-shot VQA
* Image-caption
   * Fine-tuned on COCO
   * LLM is frozen and Q-former and image encoder are finetuned
* Visual Question Answering
   * Fine-tuned on COCO
   * LLM is frozen and Q-former and image encoder are finetuned
* Image-text retrieval
   * Fine-tuned on COCO
   * Only Q-former and image encoder are finetuned

## Limitation
This is the limitation mentioned in the paper,
> Recent LLMs can perform in-context learning given fewshot examples. However, our experiments with BLIP-2 do not observe an improved VQA performance when providing the LLM with in-context VQA examples. We attribute the lack of in-context learning capability to our pretraining dataset, which only contains a single image-text pair per sample. The LLMs cannot learn from it the correlation among multiple image-text pairs in a single sequence.

But I also think that the Q-former is still complex and needs to be trained with large amount of data.

# LLaVA: Visual Instruction Tuning[12]

## Motivation

In LLM, different tasks can be expressed with language instruction and LLM can solve the task by following the instruction. Those instruction tuning is text-only. LLaVA is trying to extend this to the vision-language tasks. In BLIP, BLIP-2, the task is largely limited to the description of the image. With instruction tuning of the visual language model, it can solve more tasks.

## Dataset

This is the major contribution of LLaVA. Visual instruction following dataset is rare. They use GPT-4 to generate the visual instruction following dataset.

Since image-text pairs are a common dataset, e.g. COCO, a simple way to construct the visual instruction following dataset is to use the text description of the image as the instruction

```
Human:<Question to describe the image><Image><stop>
Assistant:<image caption><stop>
```

But they found that this format is not working well. So to enrich the instruction following dataset, they use GPT-4/ChatGPT to generate 3 kinds of instructions about the image.

1. Conversation
2. Detailed description
3. Complex reasoning

The prompt for GPT is using image caption and bounding box locations(which exists in COCO dataset) to generate the instruction.

Totally 158K unique language-image instruction-following samples in total, including 58K in conversations, 23K in detailed description, and 77k in complex reasoning, respectively.

## Model Architecture

They have a vision encoder and a large language model. They vision features via the vision encoder is linearly projected to the text embedding space. Then the text instruction is added. The prediction is on the assistant's response.

* Vision encoder: CLIP vision encoder ViT-L/14
* LLM: Vicuna

### First stage pre-training

Since the vision encoder and LLM are not aligned, in the first stage, the goal is to learn the linear projection layer mentioned above to align the image features with the text embedding. They use a much larger dataset from CC3M, 595K image-text pairs. The chat template is 

```
Human:<Question to describe the image><Image><stop>
Assistant:<image caption><stop>
```

For more details about supervised fine-tuning, I have a [blog post](https://liyuan24.github.io/writings/supervised_fine_tuning.html) about it.

Both the vision encoder and LLM are frozen in this stage.

### Second stage pre-training

In this stage, the vision encoder is frozen and the LLM and the linear projection layer are finetuned. On the dataset generated by GPT-4, the chat template is

```
system-message <STOP>  
Human : X1  instruct <STOP> Assistant: X1  a <STOP>  
Human : X2  instruct <STOP> Assistant: X2  a <STOP> · · ·
```

## Evaluation

They use GPT-4 as the judge to evaluate the response of LLaVA and compare with other models, e.g. GPT-4, BLIP-2, and OpenFlamingo.

## Limitation

1. There is no common benchmark results, e.g. VQA, Image Captioning, etc. Not sure how the performance of LLaVA compares with other models.
2. Instruction following is judged by GPT-4 which has no objective metrics.

# Reference
\[1\]: @misc{radford2021learningtransferablevisualmodels,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.00020}, 
}

\[2\]: @misc{he2020momentumcontrastunsupervisedvisual,
      title={Momentum Contrast for Unsupervised Visual Representation Learning}, 
      author={Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
      year={2020},
      eprint={1911.05722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1911.05722}, 
}

\[3\]: @misc{chen2021maskedautoencoders,
      title={Masked Autoencoders Are Scalable Vision Learners}, 
      author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollar and Ross Girshick},
      year={2021},
      eprint={2111.06377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.06377}, 
}

\[4\]: @misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385}, 
}

\[5\]: @misc{dosovitskiy2020image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929}, 
}

\[6\]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

\[7\]: @misc{flamingo2022flamingo,
      title={Flamingo: a Visual Language Model for Few-Shot Learning}, 
      author={Aravind Rajeswaran and Keren Frenkel and Aviral Kumar and John Canny and Tomaso Poggio and James Zou},
      year={2022},
      eprint={2204.14198},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.14198}, 
}

\[8\]: @misc{jaegle2021perceivergeneralperceptioniterative,
      title={Perceiver: General Perception with Iterative Attention}, 
      author={Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
      year={2021},
      eprint={2103.03206},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.03206}, 
}

\[9\]: @misc{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Jinghuan Shang and Jingfeng Wang and Caiming Xiong and Steven C.H. Hoi},
      year={2022},
      eprint={2201.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.12086}, 
}

\[10\]: @misc{devlin2019bert,
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}, 
      author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
      year={2019},
      eprint={1810.04805},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1810.04805}, 
}

\[11\]: @misc{li2023blip2,
      title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
      author={Junnan Li and Jinghuan Shang and Jingfeng Wang and Caiming Xiong and Steven C.H. Hoi},
      year={2023},
      eprint={2301.12597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.12597}, 
}

\[12\]: @misc{liu2023visualinstructiontuning,
      title={Visual Instruction Tuning}, 
      author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
      year={2023},
      eprint={2304.08485},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.08485}, 
}