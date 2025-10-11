---
layout: post 
title: A Brief History of Alignment from Human Preferences
date: 2025-10-10
excerpt: From mastering Atari games to powering ChatGPT, the journey of aligning AI with human preferences has been remarkable. This post traces how preference learning evolved from game environment to the practical and powerful large language models.
---

# Human Preference Alignment

Aligning AI system with human preferences is an very interesting topic and I want to talk about it in this post.

I will mainly talk about a few foundational papers on this topic

1. 2017: Deep reinforcement learning from human preferences[1]
2. 2020: Learning to summarize from human feedback[2]
3. 2021: WebGPT: Browser-assisted question-answering with human feedback[3]
4. 2022: Training language models to follow instructions with human feedback[4]

We can see that human preferences were used in pure Reinforcement Learning(RL) in the sythetic environment like Atari games as the reward signals, then used in single Natural Language Processing task like summarization, simple agent task like web browsingand finally used in multi and importantly very general tasks like chatbots that can answer you any questions(although may not always as good as you want).

# Deep reinforcement learning from human preferences[1]

If I can only use one word to highlight this paper, it is **practicality**. The goal is to apply RL in many real-world tasks instead of just game environments. What is the key aspects of the game environment in terms of RL? **Well-defined reward signals**. We all know that AlphaGo[5] and AlphaZero[6] are two of the most successful applications of RL in game environments. And AlphaGo and AlphaZero are both based on the well-defined reward signals, whether it is the win or lose. Although very successful, it has two major limitations:
1. It relies on the well-defined reward signals, which in many real-world tasks are not well-defined.
2. Cannot generalize to other tasks. AlphaGo and AlphaZero can only play Go and cannot even play other games and needless to say do well in many real-world tasks.

In [1], the authors want to address the first limitation. For many real-world tasks, the reward signals are not clearly defined. And we know that reward is very important in RL. So the authors propose
1. First train a reward model from human feedback
2. Use the trained reward model to guide the RL training

For many tasks, although humans don't know how to solve the tasks, but they can tell which one is better meaning that humans can provide feedback for the tasks either the final result or the intermediate steps. And since human can make mistakes, when choosing what kind of feedback to use, we should let human feedback provider to give as simple as possible feedback. In this paper, the authors chose to use **comparison** between 2 video clips as the feedback, A is better than B or B is better than A. Simple and effective. The authors also talked about letting human reviewers give scores as feedback, but they found that scores are less consistent than comparison.

For more technical details about how to train the reward model and how to optimize RL against the reward model, I highly recommend reading the original paper.

## Approaches
I want to highlight the approaches in this paper.

1. They applied Deep Learning to both the reward model and the RL policy. Please note this is a 2017 paper.
2. They used human preferences to train a reward model which will be used as the reward signals in RL training.

## Tasks
When we read papers, we should pay attention to what tasks are used to demonstrate the effectiveness of the proposed method. That can illustrate the effectivess of the method but also the limitation of the method. In this paper, the authors used 2 tasks, one is Atari games and the other is MuJoCo Simulated Robot, synthetic environments. We should note that although in those environments, usually the reward signals are well-defined, but the authors intentially not use them and instead use pure human preferences to train the reward model which will be used as the reward signals in RL training. And they compare this method with the built-inreward from the synthetic environments to illustrate the effectiveness of the proposed method.

But I think this is also the limitation of the work. The goal is to apply RL in real-world examples but they used game-like tasks to demonstrate the effectiveness of the proposed method.

## Reward shaping
In games, the reward is often at the end of the game. But in this case, human can provide feedback for the intermediate steps. And this is a *shaped reward*. Very interestingly, the authors found that for some games, human can provide better reward signals than the built-in reward from the synthetic environments. And in those games, the RL policy trained with human preferences can achieve better performance.

## Final Remarks
This is a very foundational paper on using human preferences(comparison) to train a reward model and further used in RL training. Although the tasks used in the paper are game-like tasks, but the idea is very general and can be applied to many real-world tasks especially when we cannot find a good way to define the rewards. And since the training data of the reward model is from human preferences, it is also a good way to align the RL policy with human preferences.

# Learning to summarize from human feedback[2]

The core idea of this paper is very similar to [1]. The task is **summarization** which is much more practical than the game-like tasks. Very nice! It is really hard for humans to define a good reward for summarization. Summarization is a traditional NLP task and has been studied for a long time. Before the popularity of human preference alignment, the most common metric is to use ROUGE[7] to evaluate the summarization quality. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is an automatic evaluation metric that measures the overlap between generated summaries and reference summaries using n-gram matching. While useful for automated evaluation, ROUGE has limitations in capturing semantic quality and human preferences. You can see the limitation. Two summarizations can both be very high quality but it is possible that they don't have much overlap in terms of n-gram matching. So if the model generated summarization doesn't have much overlap with the reference summary, it is not necessarily a bad model. But what we really care about is whether the summarization is good in terms of human preferences instead of overlapping with the reference summary. So ROUGE is NOT aligned with human preferences.

For summarization, it is relatively easy for humans to tell which one is better. And which summarization is better also refelct the human value which will be used to align the summarization model with human preferences.

## Approaches
They used pre-trained GPT-3 as the starting point. First use Supervised Fine-tuning(SFT) to train the model to do summarization on Reddit TLDR dataset[8]. Although the base model GPT-3 is very powerful to generate text, but in terms of high quality summarization, it is not very good. So SFT here is to let the model gain the ability to do good summarization.

Then the SFT model is used to generate multiple summarization for the same article and let human choose which one is better. As mentioned in [1], this comparison data will be used to train the reward model. Then the reward model will be used to guide the RL training to do even better summarization. Notice that this is a repeated steps, model generate summarizations -> human feedback collection -> reward model training -> RL training.

## Final Remarks
This is a very early paper on using human preferences to train a reward model and further used in RL training based on a pre-trained language model. And the approach is still very popular today. And we can see that **with a pre-trained language model, RL can be applied to much more practical tasks**.

# WebGPT: Browser-assisted question-answering with human feedback[3]

In this paper, the researches let the language model to learn to browse the web and answer the questions. As the name suggests, the work also utilize a pre-trained language model as the starting point. Fine-tuning a good pre-trained model to learn to browse the web is much easier than training a model from scratch to do that. In this work, human preference data are used to let the model learn the right **actions** to take when browsing the web. This is really similar to what we want to LLM-based agents to do nowaday.

## Approaches
As mentioned above, the core idea is to let the model learn the right actions. And those are the actions that are focused in this paper,

1. Search <query> Send <query> to the Bing API and display a search results page 
2. Clicked on link <link ID> Follow the link with the given ID to a new page 
3. Find in page: <text> Find the next occurrence of <text> and scroll to it
4. Quote: <text> If <text> is found in the current page, add it as a reference 
5. Scrolled down <1, 2, 3> Scroll down a number of times 
6. Scrolled up <1, 2, 3> Scroll up a number of times 
7. Top Scroll to the top of the page 
8. Back Go to the previous page 
9. End: Answer End browsing and move to answering phase 
10. End: <Nonsense, Controversial> End browsing and skip answering phase

## Demonstration Data

They collect data from human actions when browsing the web to answer the questions. This data is what they called **demonstration data**. And this will be used to do the supervised fine-tuning(SFT) to let the model learn the right actions.

## Comparison Data
Humans will provide comparison between two answers to the question after the model browse the web. This data is what they called **comparison data**. And this will be used to train the reward model. And the reward model will be used to guide the RL training to better browse the web and answer the questions.

## Final Remarks
We can see that the task used in this paper is also very practical.

# Training language models to follow instructions with human feedback[4]
OK, finally we arrived at this very famous paper as chatGPT is based on this work. I would say the core idea of the reinforcement learning from human feedback(RLHF) in this work is not new. As we can see it has already been applied in many tasks, summarization, web browsing, etc. But this work extend the RLHF to a set of tasks which are **much more general**. Previous works focused more on one task at a time. But in this work the tasks are including generation, summarization, classification and many others.

## Demonstration Data abd Comparison Data Collection
They hired a team of 40 contractors to collect the demonstration data and comparison data. Demonstration data is for a given prompt, the human will provide the completion. Comparison data is for a given prompt, the human will provide the comparison between multiple completions. The prompts are not only summarization, and span over many tasks.

The demonstration data will be used in SFT and the comparison data will be used in training the reward model.

The prompt in demonstration data is from user submitted to the OpenAI API and contractors. And the prompt used in RL training is only from API.

## Approaches
As mentioned in [2], the first step is to do SFT to let the model do better than the pre-trained model on the cared tasks. Then the SFT model will be used to initialize the reward model. And then the reward model will be used in RL training to align the model with human preferences. Very similar to [2], but applied to a more broader set of tasks(prompts).

## Final Remarks
It is really great to see that RLHF is applied to a much broader set of tasks and it changed the world!

# Some final thoughts
Human preferences can be used as the reward signals in RL training when we cannot find a good way to define the rewards. This makes the RL applicable to many real-world tasks. But we should notice that before the pre-trained model is very powerful, the tasks used are still very limited to the synthetic environments. As mentioned in [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html), RL suffers from a lot of problems, like sample efficiency and generalization. Especially in InstructGPT[4] paper, we can see the RLHF generalizes really well, a small amount of training data making the model very useful in everyday human uses. Also mentioned in [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html), 
> Good priors could heavily reduce learning time

The pre-trained model like GPT-3 provides a very good prior for the RL training and I think that is one of the reasons why the application of RLHF to it can generalize so well.





# References

[1] Christiano, P. F., et al. (2017). Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30.

[2] Stiennon, N., et al. (2020). Learning to summarize from human feedback. *Advances in Neural Information Processing Systems*, 33, 3008-3021.

[3] Nakano, R., et al. (2021). WebGPT: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*.

[4] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730-27744.

[5] Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.

[6] Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.

[7] Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text summarization branches out*, 74-81.