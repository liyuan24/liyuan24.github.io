---
layout: post 
title: My Take on the Second Half of AI
date: 2025-04-23
excerpt: Read the second half of AI multiple times and summarize something...
---

I read [The Second Half](https://ysymyth.github.io/The-Second-Half/) multiple times. It is really a great blog talking about the frontier of the RL in LLM, thinking, reasoning and agent. Just summarize my current takes on it here. They might be wrong of course.

# Prior
Prior is something I was missing before I read this blog. I am really sorry about that since prior seems really important for RL agent to learn effectively and efficiently.
Previously I was playing with some simple games like [Pong](https://liyuan24.github.io/writings/policy_gradient_and_ppo.html) and there is no prior involved. But for complex problems, prior is really important. For example, in [CALM](https://arxiv.org/abs/2010.02903), the RL agent was trained to solve the text-based game. Since there are a lot of actions to choose from, e.g., any combination of the words,
it is really hard to learn. CALM was using a then SOTA model, GPT-2, to generate the top possible actions and then used those filtered actions to train the RL agent.

Why can GPT2 can generate the top possible actions? It is because it is a large language model and has distilled commomsense from the training data(the internet), so given the context, it can generate
reasonable actions instead of random guesses.

But we need to notice that the RL agent trained in CALM doesn't have ability to generalize to other games that it is not trained on. 

# RL generalize by LLM reasoning
LLM can generate great prior for chatting, but not that good for web browsing and other tool uses, because the training data of LLM is internet text. The web navigation and other tool use needs training data
with different distribution. But reasoning seems a universal and general ability! With the test time compute to generate reasoning tokens, the LLM can create better prior for different domains.

# Recipe
So the recipe is adding reasoning/thinking to the action space of RL agent. This will help RL agent to generalize much better.



