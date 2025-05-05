---
layout: post 
title: Reward Model
date: 2025-05-04
excerpt: Reward models are widely used in ranking LLM responses and preference alignment...
---

In the context of Large Language Models (LLMs), reward models are widely used in ranking LLM responses[1], e.g. solving the math problem, and preference alignment[2] e.g. Reinforcoement Learning from Human Feedback(RLHF). In this blog, I am going to talk about 3 reward models

1. Outcome Reward Model(ORM): for ranking the math solutions by predicting reward of the final answer
2. Process Reward Model(PRM): for ranking the math solutions by predicting reward for each reasoning step
3. Bradley-Terry Model: for human preference alignment

The implementation of them could be found at my [minRewardModel](https://github.com/liyuan24/minRewardModel) repo.

# Outcome and Process Reward Model
The main reference is the *Let's Verify Step by Step* paper[1] from OpenAI. Before we talking about the model, I want to say that this is a really great paper in that it has a lot of details about how to collect the training data and maximize the value of human annotation.

The outcome reward model and process reward model are used to ranking the LLM answers for a math problem. The answer with highest score will be selected as the final answer of the LLM.

## Why we need reward model
I think it is necessary to talk about this before we dive into the model. The Chain-of-Thought(CoT)[3] prompting technique is a great way to encourage the LLM to solve a problem step by step so that the accuracy of the final answer is higher. Based on CoT, Self-Consistency[4] is proposed to further improve the quality of the CoT by letting LLM generate multiple CoT solutions for a problem and then use the majority vote to select the final answer.

Reward model is not a prompting technique. We need to collect training data of math questions and step-by-step answers pairs. Then we will train a model to predict whether an LLM answer is correct or not in the sense of final answer or each step. So it is expected to have higher accuracy than the pure prompting techniques.

## ORM
* Model: a language model with a binary classification head
* Input: a math problem and a LLM answer
* Label: a binary label(correct or incorrect of the final answer)

The label will be used to supervise the output of the ORM **per token**. The training process is very similar to next-token prediction LLM. The only difference is that the label is not the next token, but the correctness of the final answer. Note that an alternative is to only supervise on the final token. Leave this as an exercise.

When doing the inference, only the final token will be used to predict the correctness of the answer.

### Pros and Cons

The advantage of ORM is that the label is easy to collect since the final answer is the only thing that matters. The disadvantage is that with this supervision, there would be false positive meaning that the final answer is correct but some reasoning steps might be incorrect.

## PRM
* Model: a language model(**no classification head**)
* Input: a math problem and a LLM step-by-step answer
* Label: a binary label(correct or incorrect) for each step

It is most interesting that PRM converts a classification problem into an auto-regression problem. More specifically, for an step-by-step answer, we will add a special token to the end of each step. This special token will be used to locate the position of the per-step prediction. For example, we have a question and a step-by-step answer as follows:

```
The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term? Express your answer as a common fraction.
Step: 1: To find the fifth term, I need to identify the common difference of the arithmetic sequence and add it to the fourth term.
Step: 2: The common difference is the same for any consecutive pair of terms, so I can use any of them to find it.
Step: 3: For example, using the first and second terms, I can write $x-y = x+y + d$, where $d$ is the common difference.
Step: 4: Solving for $d$, I get $d = -2y$.
Step: 5: Using another pair of terms, such as the second and third, I can check if this value of $d$ is consistent.
Step: 6: I have $xy = x-y + d$, so substituting $d = -2y$, I get $xy = x-y - 2y$.
Step: 7: Simplifying, I get $xy = x - 3y$.
Step: 8: This seems like a reasonable equation, so I will assume that $d = -2y$ is correct.
Step: 9: Now, to find the fifth term, I need to add $d$ to the fourth term.
Step: 10: The fourth term is $x/y$, so the fifth term is $x/y + d = x/y - 2y$.
Step: 11: To express this as a common fraction, I need to find a common denominator for $x/y$ and $-2y$.
Step: 12: The least common denominator is $y$, so I can multiply the numerator and denominator of $-2y$ by $y$ to get $-2y^2/y$.
Step: 13: Therefore, the fifth term is $x/y - 2y^2/y = (x - 2y^2)/y$.

# Answer

(x - 2y^2)/y
```

The special token, e.g.`<|prm_label|>`, will be added to the end of each step, so the input is pre-processed to be:

```
The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term? Express your answer as a common fraction.
Step: 1: To find the fifth term, I need to identify the common difference of the arithmetic sequence and add it to the fourth term.<|prm_label|>
Step: 2: The common difference is the same for any consecutive pair of terms, so I can use any of them to find it.<|prm_label|>
Step: 3: For example, using the first and second terms, I can write $x-y = x+y + d$, where $d$ is the common difference.<|prm_label|>
Step: 4: Solving for $d$, I get $d = -2y$.<|prm_label|>
Step: 5: Using another pair of terms, such as the second and third, I can check if this value of $d$ is consistent.<|prm_label|>
Step: 6: I have $xy = x-y + d$, so substituting $d = -2y$, I get $xy = x-y - 2y$.<|prm_label|>
Step: 7: Simplifying, I get $xy = x - 3y$.<|prm_label|>
Step: 8: This seems like a reasonable equation, so I will assume that $d = -2y$ is correct.<|prm_label|>
Step: 9: Now, to find the fifth term, I need to add $d$ to the fourth term.<|prm_label|>
Step: 10: The fourth term is $x/y$, so the fifth term is $x/y + d = x/y - 2y$.<|prm_label|>
Step: 11: To express this as a common fraction, I need to find a common denominator for $x/y$ and $-2y$.<|prm_label|>
Step: 12: The least common denominator is $y$, so I can multiply the numerator and denominator of $-2y$ by $y$ to get $-2y^2/y$.<|prm_label|>
Step: 13: Therefore, the fifth term is $x/y - 2y^2/y = (x - 2y^2)/y$.

# Answer

(x - 2y^2)/y<|prm_label|>
```

During the training, it is the auto-regression process like the language model training. The differences are
1. The token id of the special token will be used to locate the position of the per-step prediction.
2. The output for the special token is a vector of lenght vocabulary size. Since this is a binary classification, we need to find 2 token positions to be the correct and incorrect prediction score. In my implementation, I use `+` token and `-` token as the correct and incorrect prediction scores.
3. After getting the scores for each step, a cross entropy loss will be applied to the scores.

The inference is pretty much the same as training process. The solution level score will be the product of the correct scores of each step.

### Pros and Cons

The advantage of PRM is that it is more fine-grained than ORM. The disadvantage is that the label is much harder to collect since we need to annotate for each step.

# Bradley-Terry Model

The Bradley-Terry model is a model for pairwise comparison. It is a great model for human preference alignment.

* Model: a language model with a regression head
* Input: a human input and a pair of chosen and rejected LLM answers
* Label: which response is chosen and which is rejected

To compare two responses, `response_1` and `response_2`, the reward mdoel will output rewards `r_1` and `r_2` for them. Then the probability of `response_1` being chosen over `response_2` is given by:

$$
P(response_1 \text{ is chosen over } response_2) = \frac{e^{r_1}}{e^{r_1} + e^{r_2}}
$$

If the ground-truth preference is $response_1 \succ response_2$, then the label is `1`. And the loss function would be:

$$
L = - \log \frac{e^{r_1}}{e^{r_1} + e^{r_2}}
$$

So we successfully convert a pairwise comparison problem into a classification problem.

Note that if the ground-truth preference is $response_2 \succ response_1$, then the label is not `0`, but we will change the loss function to be the following and the label is still `1`.

$$
L = - \log \frac{e^{r_2}}{e^{r_1} + e^{r_2}}
$$

# Limitation

Before we wrap up, I want to talk about the limitation of the reward model. As mentioned in Lilian Weng's [Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), sometimes the desired behavior is hard to be captured by the reward model. Math problem is easy in this sense since the final output or per step is relatively clear and easy to model. But there are a lot of real scenarios where larger reward doesn't always lead to behavior we want.


# References
\[1\]: @misc{lightman2023letsverifystepstep,
      title={Let's Verify Step by Step}, 
      author={Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
      year={2023},
      eprint={2305.20050},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20050}, 
}

\[2\]: @misc{ouyang2022traininglanguagemodelsfollow,
      title={Training language models to follow instructions with human feedback}, 
      author={Long Ouyang and Jeff Wu and Xu Jiang and Diogo Almeida and Carroll L. Wainwright and Pamela Mishkin and Chong Zhang and Sandhini Agarwal and Katarina Slama and Alex Ray and John Schulman and Jacob Hilton and Fraser Kelton and Luke Miller and Maddie Simens and Amanda Askell and Peter Welinder and Paul Christiano and Jan Leike and Ryan Lowe},
      year={2022},
      eprint={2203.02155},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.02155}, 
}

\[3\]: @misc{wei2023chainofthoughtpromptingelicitsreasoning,
      title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
      author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
      year={2023},
      eprint={2201.11903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2201.11903}, 
}

\[4\]: @misc{wang2023selfconsistencyimproveschainthought,
      title={Self-Consistency Improves Chain of Thought Reasoning in Language Models}, 
      author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc Le and Ed Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
      year={2023},
      eprint={2203.11171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.11171}, 
}