---
layout: post 
title: Policy Gradient and PPO from Scratch
date: 2025-04-19
excerpt: Let's implement Policy Gradient and PPO from scratch...
---

Reinforcement Learning(RL) is really popular recently because the reasoning ability of the advanced LLM like DeepSeek R1 and GPT o1/2/3/4 is enhanced by reinforcement learning training.
That is also the reason why I want to learn more about the reinforcement learning. This blog is the first part of the reinforcement learning series and we will talk about the Policy Gradient[1]
and Proximal Policy Optimization(PPO)[2] and build them from scratch.

<div style="text-align:center;">
<iframe width="420" height="315" src="https://www.youtube.com/embed/gvx3N3UDTuc?autoplay=1&amp;loop=1&amp;rel=0&amp;showinfo=0&amp;playlist=gvx3N3UDTuc" frameborder="0" allowfullscreen></iframe>
<br>
The learned agent is on the right.
</div>

The agent is trained by PPO on one GTX 3090 for about 2 hours.

# RL Quick Primer
In RL, we have an agent that is interacting with the environment. There are some key concepts here

- State: The state of the environment at a certain time step. This is the input information for the agent to make a decision.
  - For language model, as it will predict the next token, the state is the tokens in the history.
- Action: Based on the state, the agent will take an action at a certain time step.
  - For language model, the action is the next token in the vocabulary.
- Reward: The reward received by the agent at a certain time step after taking the action.
  - For language model, the reward can be *rule* based(e.g.solving the math problem correctly) or a *reward model* based(e.g. whether the LLM response is aligned with the human preference)
- Policy: The policy is the mapping from the state to the action.
  - For language model, policy is the language model itself, yay!

So **the goal of the agent is to learn a policy that can maximize the expected reward**.

> **Note:** I read [The Second Half](https://ysymyth.github.io/The-Second-Half/) from Yao Shunyu yesterday. It mentioned that **prior** is a very important component in RL. Please take a look if you are
> interested in the state of the art of RL.

# Credit Assignment Problem
As mentioned above, the goal of the agent is to learn a policy that can maximize the expected reward. For example, in the language model, when the LLM is prompted to solve a math problem, it will output the response word by word(more precisely, token by token). 
There is no reward until the LLM completes the response. Or in the [Pong game](https://ale.farama.org/environments/pong/), the agent can move the paddle up or down to hit the ball. There is no reward until the end of the game. That means that the agent can take a lot of actions before getting the reward signal to evaluate whether the action is good or not. You can see the problem here: **in the middle of the LLM response or the Pong game, how do we evaluate the quality of the token or the UP/DOWN action?**

This is the credit assignment problem. And here I mention 3 ways to solve this problem:

1. Monte Carlo Sampling
2. Temporal Difference Learning
3. Generalized Advantage Estimation

## Monte Carlo(MC) Sampling
Monte Carlo Sampling is the simplest way to solve the credit assignment problem. The idea is to wait until the **end of the game** and use the final total rewards to assign the credit the all actions in the history. For example, in the Pong game, if the agent wins the game and get score 1, we will assign the credit 1 to all the actions in the history. And if the agent loses the game, we will assign the credit -1 to all the actions in the history. We could also have a **discount factor** to introduce the impact of the time.
The reward assigned to action at timestep `t` is:

$$
\begin{align}
R_t = \sum_{i=t}^{T} \gamma^{i-t} r_i \tag{1}
\end{align}
$$

where $\gamma$ is the discount factor between 0 and 1 and $r_i$ is the reward at timestep `i`.

I want to highlight some pros and cons of Monte Carlo Sampling:

- **Pros**:
  - Simple and easy to understand
  - The credit assignment is unbiased as we **only** use the the real rewards.
- **Cons**:
  - Need to wait until the end of the game which means the agent cannot have quick feedback.
  - Since both the environment and policy are stochastic, the final reward is very noisy. This will lead to high variance of the reward signal and slow learning. More in the next section on *Bias and Variance Trade-offs in RL*.

## Temporal Difference(TD) Learning
OK, so in MC, the real rewards are used but we need to wait until the end of the game. The question is *can we have an estimate of the rewards without waiting until the end of the game?* The answer is yes!
And we need to introduce some model to estimate the value of the next state. In another word, when the agent takes an action $a_t$ at timestep `t`, it gets a reward $r_t$ which could be 0 and goes into 
the next state $s_{t+1}$, but since the
game is not over, the agent's action $a_t$ cannot be evaluated immediately. So we need a way to estimate the value of the next state $s_{t+1}$ and use it to evaluate the action $a_t$. This is the
the idea of TD learning. The credit assigned to action $a_t$ is:

$$
R_t = r_t + \gamma V(s_{t+1}) \tag{2}
$$

where $V(s_{t+1})$ is the estimated value of the next state $s_{t+1}$ and $\gamma$ is the discount factor.

And in practice we will consider the *advantage* of the action $a_t$ compared to the average reward of the state $s_t$.

**Advantage**[1] is a very important and interesting concept in RL. It is defined as 

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \tag{3}
$$

where $Q(s_t, a_t)$ is the expected reward of taking action $a_t$ at state $s_t$ and $V(s_t)$ is the estimated value of state $s_t$. 

Intuitively, we can think of it as how much better the action $a_t$ taken at state $s_t$ is compared to the average reward of the state $s_t$. $V(s_t)$ can be seen as a *baseline* of state $s_t$.
That makes sense right? The absolute value of the credit assigned to action $a_t$ is not important, but the relative value compared to the baseline is what matters. Like in the company performance review,
the absolute impact you have made in the past year is not that important, what matters is how you compare to the average performance of the company or the org.

So in TD learning, we will compute the advantage as follows:

$$
A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{4}
$$

This is also called $TD(0)$ error. We use both the real reward $r_t$ and the estimated value function $V(s_t)$ and $V(s_{t+1})$ to estimate the advantage.

## Generalized Advantage Estimation(GAE)[3]

GAE is a more general way and use a parameter $\lambda$ to control the mix of real rewards and the estimated value function. Before introducing GAE, let's first introduce the n-step TD error, $TD(n)$:

$$
TD(n) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^{n} r_{t+n} + \gamma^n V(s_{t+n+1}) - V(s_t) \tag{5}
$$

You can see that $TD(n)$ error uses more real rewards that the $TD(0)$ error. And if $n->\infty$, $TD(n)$ error will be MC as in equation (1).

Back to GAE, it introduces a parameter $\lambda$ to control the mix of real rewards and the estimated value function. The credit assigned to action $a_t$ is:

$$
\begin{align}
A(s_t, a_t) &= (1-\lambda) \sum_{n=0}^{\infty} \lambda^{n} TD(n) \\
            &= \sum_{n=0}^{\infty} (\lambda\gamma)^{n} \delta_t
\end{align}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the $TD(0)$ error at timestep `t`.

When $\lambda = 0$, GAE is the same as $TD(0)$ error as in equation (4). When $\lambda = 1$, GAE is the same as MC as in equation (1).

# RL vs Supervised Learning

We talked about how to do the credit assignment in the previous section. Now I want to talk a little bit about the comparison between RL and supervised learning. In supervised learning like image classification,
whether the image is a cat or not, the label is very clear. But in RL, there is **no label, only reward**. How the agent adjust its policy(action) to earn more rewards is kind of supervised
by the credit assigned to the actions and we'll see how it manifests in the policy gradient and PPO algorithms.

And that is why the credit assignment is the key problem in RL.

For more read about this topic, I recommend reading the Policy Gradient [blog](https://karpathy.github.io/2016/05/31/rl/) from Andrej Karpathy

> **Note:** The loss function of RL is different from supervised learning in that for supervised learning, the loss funciton
> can never be negative. But in RL, it could be negative.

# Bias and Variance Trade-offs in RL
OK before we dive into the policy gradient and PPO algorithms, please allow me to cover the bias and variance trade-offs in RL which I think is an very interesting topic. In supervised learning,
we also have the bias and variance trade-offs. But it has different meaning as in RL. For example, in the image classification, if the model is too simple, it will be underfitting. We say in this
case the model has *low variance but high bias*. On the other hand, if the model is too complex, it will be overfitting and we say the model has *high variance but low bias*.

In RL, the bias and variance trade-offs are about reward signal collection. In the Monte Carlo Sampling, we use the real rewards and wait until the end of the game to collect all the reward signals. Since both the environment and policy are stochastic,
the final reward is very noisy. That means the reward signal has *low bias but high variance*. On the other hand, in the TD learning, we use a model to estimate the value function.
Since estimated value function is more stable, in this case the reward signal has *low variance but high bias*. The high bias is due to the fact that the estimated value function
is not an unbiased estimator of the real value function.

You might ask although MC has high variance, but it is unbiased, why do people bother with TD learning? The reason is that the high variance leads to very slow learning[1] since it needs a lot samples
to train the model to make the law of large numbers take effect. An intuition is that an action can sometimes lead to a very high reward and sometimes lead to a very low reward(high variance) which makes the agent really
confused and slow to learn.

# Policy Gradient(PG)[1]
Policy gradient is a very popular family of the RL algorithms. It has many variants, e.g., PPO(next section), Group Relative Policy Optimization(GRPO)[4], etc. PG is directly modeling the policy function. Remember that
the goal of the RL agent is to learn a policy that can maximize the expected reward. And PG is directly targeting the policy which is great. Before I wrote this blog, I thought this is an algorithm that is probably 1 or 2
decade old. But actually [1] was first published in 1999.

So the idea is to have a model(e.g. a neural network) to parameterize the policy function and then use the gradient of the mdoel to iteratively update the model with gradient ascent or descent(depending on the loss function).

Before we dive into the code, I want to first lay out the theoretical foundation of PG.

## Policy Gradient Theorem
Let's define the policy function model as $\pi(a|s;\theta)$ where $\theta$ is the parameter of the model. The goal of the RL agent is to learn a policy that can maximize the expected reward and mathematically
we could express it as
$$
J(\pi(\theta)) = \mathbb{E}_{\tau\sim\pi(\theta)}[R(\tau)]
$$,
where 
1. $\tau$ is trajectory of the agent, which is a sequence of states and actions, $\tau = (s_0, a_0, s_1, a_1, s_2, a_2, \cdots, s_T, a_T)$, $s_t$ is the state at timestep `t` and $a_t$ is the action at timestep `t`
2. $R(\tau)$ is the total reward of the trajectory $\tau$

What policy gradient does is to get the gradient of the total expected reward with respect to the policy parameter $\theta$ and use it to update the policy parameter. So let's derive it.

$$
\begin{align}
\nabla_\theta J(\pi(\theta)) &= \nabla_\theta \mathbb{E}_{\tau\sim\pi(\theta)}[R(\tau)] \\
                             &= \nabla_\theta \int_{\tau} P(\tau;\theta) R(\tau) d\tau \\
                             &= \int_{\tau} \nabla_\theta (\sum_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi(a_t|s_t;\theta)) R(\tau) d\tau \\
                             &= \int_{\tau} \nabla_\theta (\sum_{t=0}^{T} \pi(a_t|s_t;\theta)) R(\tau) d\tau \\
                             &= \int_{\tau} (\sum_{t=0}^{T} \nabla_\theta \pi(a_t|s_t;\theta)) R(\tau) d\tau \\
                             &= \int_{\tau} (\sum_{t=0}^{T} \pi(a_t|s_t;\theta)\frac{\nabla_\theta \pi(a_t|s_t;\theta)}{\pi(a_t|s_t;\theta)}) R(\tau) d\tau \\
                             &= \int_{\tau} (\sum_{t=0}^{T} \pi(a_t|s_t;\theta)\nabla_\theta \log\pi(a_t|s_t;\theta)) R(\tau) d\tau \\
                             &= \mathbb{E}_{\tau\sim\pi(\theta)}(\sum_{t=0}^{T} \nabla_\theta \log\pi(a_t|s_t;\theta) R(\tau) d\tau \\
\end{align}
$$

So in practice, if we have $N$ trajectories, we can estimate the gradient as follows:

$$
\nabla_\theta J(\pi(\theta)) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log\pi(a_t^{(i)}|s_t^{(i)};\theta) R(\tau^{(i)})
$$

## Some variants of the policy gradient
The reward $R(\tau)$ used above is the total reward of the trajectory. As we mentioned in the Variance and Bias trade-off section, it has very high variance. There are some variants which use other terms to replace
the total reward. We will not provide the rigorous proof here, if you are interested in the proof, please check
[Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#part-3-intro-to-policy-optimization) from OpenAI.

### Reward-to-go
The reward-to-go is defined as:

$$
R_t = \sum_{i=t}^{T} \gamma^{i-t} r_i
$$

And the policy gradient would be 

$$
\nabla_\theta J(\pi(\theta)) = \mathbb{E} \sum_{t=0}^{T} \nabla_\theta \log\pi(a_t^{(i)}|s_t^{(i)};\theta) R_t^{(i)}
$$

At first it really makes sense to me. The action at timestep `t` will only affect the reward after timestep `t`. So for action at timestep `t`, we only use the reward after timestep `t` to evaluate it.

How to intuitively understand this? The derivative of the log probability is the direction to **increase** the probability of the action.
So if the reward $R_t^{(i)}$ is positive, we want to increase the probability of the action $a_t^{(i)}$ and if the reward is negative, we want to decrease the probability of the action $a_t^{(i)}$.

### Advantage
Advantage is defined in equation (3).

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

where $Q(s_t, a_t)$ is the expected reward of taking action $a_t$ at state $s_t$ and $V(s_t)$ is the estimated value of state $s_t$.
It uses an idea called *baseline* to reduce the variance and keep the estimator unbiased which is amazing!

How to intuitively understand this? I want to quote what mentioned in [3]

> that a step in the policy gradient direction should increase the probability of better-than-average actions and decrease the probability of worse-thanaverage actions

## Let's build PG from scratch by playing Pong

Talk is cheap, show me the code! I implemented a *reward-to-go* version of PG using the Atari Pong game. This example is inspired by Andrej Karpathy's [blog](https://karpathy.github.io/2016/05/31/rl/).
The source code could be found at my [reinforcement_learning repo](https://github.com/liyuan24/reinforcement_learning/blob/main/vanilla_policy_gradient_parallel_envs.py).

I just want to highlight one thing here. We derived the policy gradient above and it could be used to update the policy network parameters. But in Pytorch or other deep learning frameworks, we need to have
a loss function. From the gradient, the loss function would look like

$$
loss =  -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \log\pi(a_t^{(i)}|s_t^{(i)};\theta) R(\tau^{(i)})
$$

We have a minus sign here because Pytorch uses gradient descent to update the parameters instead of gradient ascent.

# PPO

With the understanding of PG, it is easy to understand PPO. PPO is still a policy gradient method. It is concerned with
1. How to reduce the variance of the reward signal.
2. How to not over-react to the reward signal. As the reward signal sometimes can be very noisy.
3. How to make use of the training data more efficiently.

To address the **first** concern, PPO introduce a value function $V$ aka *critic* to estimate the value of the state. So there are two networks now, the *policy* network and the *value* network.
They use GAE as the reward signal. The value function network is updated together with the policy network by introducing a *value loss* function.
$$
L_{value} = \mathbb{E}_t \left[ (V(s_t) - R_t)^2 \right]
$$

To address the **second** concern, PPO uses a *clipped* surrogate objective function.

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

where 

$$
r_t(\theta) = \frac{\pi(a_t|s_t;\theta)}{\pi(a_t|s_t;\theta_{old})}
$$

It looks complicated at first glance. But the idea is to not over-react to the reward signal.
When $\hat{A}_t > 0$, if $r_t(\theta) > 1+\epsilon$, the loss will be clipped to $1+\epsilon$. What does this mean? It means the gradient will be **0**
because this objective function is a constant. So that this training example will not contribute to the gradient update. 
Similarly, when $\hat{A}_t < 0$, if $r_t(\theta) < 1-\epsilon$, the loss will be clipped to $1-\epsilon$.

To address the **third** concern, for each training data collection round, there would be multiple epochs of parameter updates with the same set of training data.

And finally, they also introduce a entropy bonus to the objective function to encourage the policy to be more explorative.

$$
L^{entropy}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \pi(a_t^{(i)}|s_t^{(i)};\theta) \log\pi(a_t^{(i)}|s_t^{(i)};\theta)
$$

So the final objective function(to be maximized) is:

$$
L(\theta) = L^{CLIP}(\theta) - c_1 L_{value}(\theta) + c_2 L^{entropy}(\theta)
$$

where $c_1$ and $c_2$ are the coefficients for the value loss and entropy bonus.

## Implement PPO from scratch by playing Pong
The source code could be found at my [reinforcement_learning repo](https://github.com/liyuan24/reinforcement_learning/blob/main/ppo.py).




# Reference

\[1\]: @inproceedings{NIPS1999_464d828b,
 author = {Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Solla and T. Leen and K. M\"{u}ller},
 pages = {},
 publisher = {MIT Press},
 title = {Policy Gradient Methods for Reinforcement Learning with Function Approximation},
 url = {https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf},
 volume = {12},
 year = {1999}
}

\[2\]: @article{DBLP:journals/corr/SchulmanWDRK17,
  author       = {John Schulman and
                  Filip Wolski and
                  Prafulla Dhariwal and
                  Alec Radford and
                  Oleg Klimov},
  title        = {Proximal Policy Optimization Algorithms},
  journal      = {CoRR},
  volume       = {abs/1707.06347},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.06347},
  eprinttype    = {arXiv},
  eprint       = {1707.06347},
  timestamp    = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

\[3\]: @misc{schulman2018highdimensionalcontinuouscontrolusing,
      title={High-Dimensional Continuous Control Using Generalized Advantage Estimation}, 
      author={John Schulman and Philipp Moritz and Sergey Levine and Michael Jordan and Pieter Abbeel},
      year={2018},
      eprint={1506.02438},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1506.02438}, 
}

\[4\]: @misc{shao2024deepseekmathpushinglimitsmathematical,
      title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}, 
      author={Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Xiao Bi and Haowei Zhang and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
      year={2024},
      eprint={2402.03300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.03300}, 
}

