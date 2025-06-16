---
layout: post 
title: Inference Time Compute
date: 2025-06-11
excerpt: Use more inference time compute to get the better performance...
---

updated[2025-06-15]: added Least-to-Most Prompting[5]

# Inference Time Compute

In this blog, I am going to talk about using more inference time compute to get the better performance. There is no post-training or fine-tuning involved, only prompting.

# Chain of Thought[1]

Chain of thought(CoT) which is a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. The authors showed that the CoT prompting with a few demonstrations can trigger the reasoning ability in sufficiently large language models.

## Motivation

1. The natural language rationale can help solve the arithmetic reasoning problem.
2. [Language models are few-shot learners](https://arxiv.org/abs/2005.14165). Basically by providing a few demonstration examples for a given task, the language model can be prompted to perform the task.

Both of them have their own limitation. For 1, to get better performance, the model needs to be fine-tuned on the natural language rationale data which are hard to collect. For 2, the performance on the complex reasoning tasks is not good.

The authors proposed to combine them with prompt format like `<input> <intermediate_steps> <output>`.

## Prompt examples

Arithmetic reasoning. The LLM is prompted with steps to solve the problem.

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. 
How many tennis balls does he have now?  

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 
5 + 6 = 11. The answer is 11.
```

## More inference time compute

As mentioned above, the LLM is prompted with steps to solve the problem. By generating more steps, more test time compute is allocated to solve the problem.

## Evaluation

They evaluate the performance of CoT on the following tasks benchmarks:

* arithmetic
* commonsense
* symbolic reasoning

# Self-consistency[2]

Self-consistency is a method to improve the performance of CoT. The idea is to generate multiple CoT solutions for the same input and then select the most consistent one as the final answer. The most consistent one can be roughly seen as majority voting. To me, this is like an ensemble method but instead of from different models, it is from the same model with same prompt. But since they use sampling like temperature sample, top-p sampling, top-k sampling, etc, the diversity of the output is high.

## Motivation

The model can generate wrong answer, but different reasoning path may not generate the same wrong answer.

## Prompt examples

They use CoT prompting, so this is the same as the CoT examples above.

## More inference time compute

This is very clear, more reasoning paths, more inference time compute.

## Evaluation

The same benchmarks as CoT are used.

# Least-to-Most Prompting[5]
Least-to-Most prompting is a multi-stage prompting method. In the first stage, the problem will be prompted to be decomposed into sub-problems. In the second stage, the sub-problems will be solved one by one sequentially with the answers to the previous sub-problems in the context of the next sub-problem.

## Motivation

CoT doesn't generalize well from simple problem to hard problem. For example in the last letter concatenation problem, the CoT can do well when the test list length is the same as the few-shot examples list length. But when the test list length is longer, the performance drops.

## Prompt examples

This example is from [learnprompting.org](https://learnprompting.org/docs/intermediate/least_to_most)

```
Q: think, machine
A: The last letter of "think" is "k". The last letter of "machine" is "e". Concatenating "k" and "e" gives "ke". So "think, machine" output "ke".

Q: think, machine, learning
A: "think, machine" outputs "ke". The last letter of "learning" is "g". Concatenating "ke" and "g" gives "keg". So "think, machine, learning" is "keg".

Q: transformer, language
A: The last letter of "transformer" is "r". The last letter of "language" is "e". Concatenating "r" and "e" gives "re". So "transformer, language" is "re".

Q: transformer, language, vision
A: "transformer, language" outputs "re". The last letter of "vision" is "n". Concatenating "re" and "n" gives "ren". So "transformer, language, vision" is "ren".

Q: foo,bar,baz,blip,learn,prompting,world,shaking,event,dancefloor,prisma,giraffe
A:
```


# ReAct[3]

CoT and self-consistency are in-context learning methods. They rely on the learable weights of the language models to solve the problems. ReAct is one step further by utilizing the external tools, e.g. search engine, python code execution, web crawler, etc. What is really amazing is that you just need to provide the tool description, when to use the tool and the tool input will be figured out by LLM itself.

## Motivation

CoT and self-consistency can still have hullucination problem as they generate based on the model parameters. ReAct is trying to combine the reasoning path from CoT with the external tool use.

## Prompt examples

The following prompt is from my [nanoDeepResearch](https://github.com/liyuan24/nanoDeepResearch/blob/main/agent/react_agent.py) repo

```
Human query: {agent_input}
Follow this format:
Thought: Think about the current situation and what to do
Action: The action to take (must be one of: {', '.join([tool.name for tool in self.tools])})
Action Input: The input to the action (can be a string or a JSON object)
Observation: The result of the action
... (this Thought/Action/Observation cycle can repeat multiple times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question
```

We can see that the LLM will generate

1. The thought
2. What tool to use
3. The tool input

Then an external tool will be called to get the response which will be the observation. After that the LLM will repeat the thought, action, action input based on the observation until it finds the final answer.

## More inference time compute

With the loop of thought, action, action input, observation, the inference time compute is increased.

## Evaluation

1. KNOWLEDGE-INTENSIVE REASONING TASKS
   * HotPotQA: a multi-hop question answering benchmark that requires reasoning over two or more Wikipedia passages,
   * FEVER: a fact verification benchmark where each claim is annotated SUPPORTS, REFUTES, or NOT ENOUGH INFO, based on if there exists a Wikipedia passage to verify the claim
2. DECISION MAKING TASKS
   * ALFWorld: a synthetic text-based game designed to align with the embodied ALFRED benchmark (Shridhar et al., 2020a)
   * WebShop: an online shopping website environment with 1.18M real-world products and 12k human instructions.

# PlanSearch[4]

They define *search* as 

> any method that uses more test time compute to improve the LLM performance on the problem

In this paper, they want to use search to improve the coding performance on several benchmarks. 

## Hypothesis 

The repeated sampling doesn’t generate diverse responses because of the post-training optimizing for generating single best response.
So they want to use search to increase the diversity. But the question is what is the search space?? They hypothesize right axis of diversity to search over is the natural language conceptual/idea space,


## Evidence for the hypothesis

this is very smart
1: for a correct solution code, they backtranslate the code into idea space(which is the correct sketch) and use this sketch together with the problem as the prompt to generate the solution. They measure the pass@1 and pass@5 with varying length of natural language description and find that the performance increase as the length increases
2: for each problem, they generate the natural language description first and then compute the pass rate. Since they don’t know whether the description is good or not, they use a proxy which is the pass rate with natural language description polarizes toward 0 and 1. That means the goodness of description is important at least

## Method
So with those evidences, it is clear that the natural language description is important to improve the pass rate. Their contribution is to construct the high-quality description.

1. They use two-stage prompting, the first state generate several obs/hints in natural language for a given problem(prompt1)
2. The second stage is given the problem and the first stage description to generate new obs(prompt2)
3. They those 1st and 2nd stage descriptions are combined into a description(prompt3)
4. This description is used to generate the pseudo code 
5. Then the real code is gneereated

## Experiment

Test on several benchmarks and find that their method is the best compared to repeated sampling and idea search(1 stage)

## Filter

The idea is to use some simple tests case to filter out the incorrect solutions so that no need to submit many solutions. Make the pass rate @200 the same as passrate@20 .e.g.

## Diversity measurement

They use LLM to give a yes or no answer with the prompt of problem statement, two code solutions and two ideas(backtranslated), and ask LLM to see if the two ideas are similar(yes or no)
The diversity score for a problem is the the percentage of NO
They found that the passrate is positive related to the diversity score

## Limitation

They increase the performance with more inference time compute by searching over the idea space. But some application may not be able to afford the latency of generating a lot of solutions, such as chatbot. 

# Reference

\[1\]: @misc{wei2023chainofthoughtpromptingelicitsreasoning,
      title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
      author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
      year={2023},
      eprint={2201.11903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2201.11903}, 
}

\[2\]: @misc{wang2023selfconsistencyimproveschainthought,
      title={Self-Consistency Improves Chain of Thought Reasoning in Language Models}, 
      author={Xuezhi Wang and Jason Wei and Dale Schuurmans and Quoc Le and Ed Chi and Sharan Narang and Aakanksha Chowdhery and Denny Zhou},
      year={2023},
      eprint={2203.11171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.11171}, 
}

\[3\]: @misc{yao2023reactsynergizingreasoningacting,
      title={ReAct: Synergizing Reasoning and Acting in Language Models}, 
      author={Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and Izhak Shafran and Karthik Narasimhan and Yuan Cao},
      year={2023},
      eprint={2210.03629},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2210.03629}, 
}

\[4\]: @misc{wang2024planningnaturallanguageimproves,
      title={Planning In Natural Language Improves LLM Search For Code Generation}, 
      author={Evan Wang and Federico Cassano and Catherine Wu and Yunfeng Bai and Will Song and Vaskar Nath and Ziwen Han and Sean Hendryx and Summer Yue and Hugh Zhang},
      year={2024},
      eprint={2409.03733},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.03733}, 
}

\[5\]: @misc{zhou2023leasttomostpromptingenablescomplex,
      title={Least-to-Most Prompting Enables Complex Reasoning in Large Language Models}, 
      author={Denny Zhou and Nathanael Schärli and Le Hou and Jason Wei and Nathan Scales and Xuezhi Wang and Dale Schuurmans and Claire Cui and Olivier Bousquet and Quoc Le and Ed Chi},
      year={2023},
      eprint={2205.10625},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2205.10625}, 
}

