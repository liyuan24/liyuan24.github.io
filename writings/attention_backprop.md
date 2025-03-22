---
layout: post 
title: Attention Backpropagation
date: 2025-03-21
excerpt: How to backpropagate the attention...
---

I recently revisited the FlashAttention[1] and FlashAttention2[2] papers. It is really fun to manually derive the backward pass of the attention.
In this blog, I will use a concrete example to illustrate this process and hope it is easy to understand.

# Forward Pass
So attention[3] involves 3 matrices: $Q$, $K$, $V$. The matrix shape is [batch_size, num_heads, seq_len, head_dim]. Attention is calculated as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{head\_dim}})V
$$

Let me use a simple example to illustrate this process. We will ignore $batch\_size$ and $num\_heads$ dimension in this example because the matrix multiplication is on $seq\_len$ and $head\_dim$ dimensions. And we will also ignore the scaling factor $\frac{1}{\sqrt{head\_dim}}$ for simplicity.

$$
Q = \begin{bmatrix}
q_{11} & q_{12} & q_{13} \\
q_{21} & q_{22} & q_{23} \\
q_{31} & q_{32} & q_{33}
\end{bmatrix}
$$

$$
K = \begin{bmatrix}
k_{11} & k_{12} & k_{13} \\
k_{21} & k_{22} & k_{23} \\
k_{31} & k_{32} & k_{33}
\end{bmatrix}
$$


$$
V = \begin{bmatrix}
v_{11} & v_{12} & v_{13} \\
v_{21} & v_{22} & v_{23} \\
v_{31} & v_{32} & v_{33}
\end{bmatrix}
$$

So 

$$
QK^T = S = \begin{bmatrix}
q_{11}k_{11} + q_{12}k_{21} + q_{13}k_{31} & q_{11}k_{12} + q_{12}k_{22} + q_{13}k_{32} & q_{11}k_{13} + q_{12}k_{23} + q_{13}k_{33} \\
q_{21}k_{11} + q_{22}k_{21} + q_{23}k_{31} & q_{21}k_{12} + q_{22}k_{22} + q_{23}k_{32} & q_{21}k_{13} + q_{22}k_{23} + q_{23}k_{33} \\
q_{31}k_{11} + q_{32}k_{21} + q_{33}k_{31} & q_{31}k_{12} + q_{32}k_{22} + q_{33}k_{32} & q_{31}k_{13} + q_{32}k_{23} + q_{33}k_{33}
\end{bmatrix} = \begin{bmatrix}
s_{11} & s_{12} & s_{13} \\
s_{21} & s_{22} & s_{23} \\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$$

$$
P = \text{softmax}(S) = \begin{bmatrix}
\frac{exp(s_{11})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} & \frac{exp(s_{12})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} & \frac{exp(s_{13})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} \\
\frac{exp(s_{21})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} & \frac{exp(s_{22})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} & \frac{exp(s_{23})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} \\
\frac{exp(s_{31})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})} & \frac{exp(s_{32})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})} & \frac{exp(s_{33})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})}
\end{bmatrix} = \begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{bmatrix}
$$

$$
O = PV = \begin{bmatrix}
p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31} & p_{11}v_{12} + p_{12}v_{22} + p_{13}v_{32} & p_{11}v_{13} + p_{12}v_{23} + p_{13}v_{33} \\
p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31} & p_{21}v_{12} + p_{22}v_{22} + p_{23}v_{32} & p_{21}v_{13} + p_{22}v_{23} + p_{23}v_{33} \\
p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31} & p_{31}v_{12} + p_{32}v_{22} + p_{33}v_{32} & p_{31}v_{13} + p_{32}v_{23} + p_{33}v_{33}
\end{bmatrix} = \begin{bmatrix}
o_{11} & o_{12} & o_{13} \\
o_{21} & o_{22} & o_{23} \\
o_{31} & o_{32} & o_{33}
\end{bmatrix}
$$

$O$ is the output of the attention.

# Backward Pass
When we do backward pass, the input is the partial derivative of loss with respect to $O$.

$$
\frac{\partial L}{\partial O} = \begin{bmatrix}
\frac{\partial L}{\partial o_{11}} & \frac{\partial L}{\partial o_{12}} & \frac{\partial L}{\partial o_{13}} \\
\frac{\partial L}{\partial o_{21}} & \frac{\partial L}{\partial o_{22}} & \frac{\partial L}{\partial o_{23}} \\
\frac{\partial L}{\partial o_{31}} & \frac{\partial L}{\partial o_{32}} & \frac{\partial L}{\partial o_{33}}
\end{bmatrix}
$$

When we use the deep learning framework like Pytorch, Jax, this derivative is automatically computed. And we will use this derivative to compute the gradient of $\frac{\partial L}{\partial Q}$, $\frac{\partial L}{\partial K}$, $\frac{\partial L}{\partial V}$.

## Gradient of $V$ and $P$

This is most straightforward. Remember that $O = PV$,


$$
O = PV = \begin{bmatrix}
p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31} & p_{11}v_{12} + p_{12}v_{22} + p_{13}v_{32} & p_{11}v_{13} + p_{12}v_{23} + p_{13}v_{33} \\
p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31} & p_{21}v_{12} + p_{22}v_{22} + p_{23}v_{32} & p_{21}v_{13} + p_{22}v_{23} + p_{23}v_{33} \\
p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31} & p_{31}v_{12} + p_{32}v_{22} + p_{33}v_{32} & p_{31}v_{13} + p_{32}v_{23} + p_{33}v_{33}
\end{bmatrix} = \begin{bmatrix}
o_{11} & o_{12} & o_{13} \\
o_{21} & o_{22} & o_{23} \\
o_{31} & o_{32} & o_{33}
\end{bmatrix}
$$

So for example $\frac{\partial L}{\partial v_{11}}$, it appears in the first column of $O$, so

$$
\frac{\partial L}{\partial v_{11}} = \frac{\partial L}{\partial o_{11}}\frac{\partial o_{11}}{\partial v_{11}} + \frac{\partial L}{\partial o_{21}}\frac{\partial o_{21}}{\partial v_{11}} + \frac{\partial L}{\partial o_{31}}\frac{\partial o_{31}}{\partial v_{11}}
$$

Since $o_{11} = p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31}$,

$$
\frac{\partial o_{11}}{\partial v_{11}} = p_{11}
$$

$$
\frac{\partial o_{21}}{\partial v_{11}} = p_{21}
$$

$$
\frac{\partial o_{31}}{\partial v_{11}} = p_{31}
$$

So

$$
\frac{\partial L}{\partial v_{11}} = \frac{\partial L}{\partial o_{11}}p_{11} + \frac{\partial L}{\partial o_{21}}p_{21} + \frac{\partial L}{\partial o_{31}}p_{31}
$$

So

$$
\frac{\partial L}{\partial V} = \begin{bmatrix}
\frac{\partial L}{\partial v_{11}} & \frac{\partial L}{\partial v_{12}} & \frac{\partial L}{\partial v_{13}} \\
\frac{\partial L}{\partial v_{21}} & \frac{\partial L}{\partial v_{22}} & \frac{\partial L}{\partial v_{23}} \\
\frac{\partial L}{\partial v_{31}} & \frac{\partial L}{\partial v_{32}} & \frac{\partial L}{\partial v_{33}}
\end{bmatrix} = \begin{bmatrix}
p_{11}\frac{\partial L}{\partial o_{11}} + p_{21}\frac{\partial L}{\partial o_{21}} + p_{31}\frac{\partial L}{\partial o_{31}} & p_{11}\frac{\partial L}{\partial o_{12}} + p_{21}\frac{\partial L}{\partial o_{22}} + p_{31}\frac{\partial L}{\partial o_{32}} & p_{11}\frac{\partial L}{\partial o_{13}} + p_{21}\frac{\partial L}{\partial o_{23}} + p_{31}\frac{\partial L}{\partial o_{33}} \\
p_{12}\frac{\partial L}{\partial o_{11}} + p_{22}\frac{\partial L}{\partial o_{21}} + p_{32}\frac{\partial L}{\partial o_{31}} & p_{12}\frac{\partial L}{\partial o_{12}} + p_{22}\frac{\partial L}{\partial o_{22}} + p_{32}\frac{\partial L}{\partial o_{32}} & p_{12}\frac{\partial L}{\partial o_{13}} + p_{22}\frac{\partial L}{\partial o_{23}} + p_{32}\frac{\partial L}{\partial o_{33}} \\
p_{31}\frac{\partial L}{\partial o_{31}} + p_{32}\frac{\partial L}{\partial o_{32}} + p_{33}\frac{\partial L}{\partial o_{33}} & p_{31}\frac{\partial L}{\partial o_{31}} + p_{32}\frac{\partial L}{\partial o_{32}} + p_{33}\frac{\partial L}{\partial o_{33}} & p_{31}\frac{\partial L}{\partial o_{31}} + p_{32}\frac{\partial L}{\partial o_{32}} + p_{33}\frac{\partial L}{\partial o_{33}}
\end{bmatrix}
$$

So 

$$
\frac{\partial L}{\partial V} = P^T \frac{\partial L}{\partial O}
$$

Similarly,  

$$
\frac{\partial L}{\partial P} = \begin{bmatrix}
\frac{\partial L}{\partial p_{11}} & \frac{\partial L}{\partial p_{12}} & \frac{\partial L}{\partial p_{13}} \\
\frac{\partial L}{\partial p_{21}} & \frac{\partial L}{\partial p_{22}} & \frac{\partial L}{\partial p_{23}} \\
\frac{\partial L}{\partial p_{31}} & \frac{\partial L}{\partial p_{32}} & \frac{\partial L}{\partial p_{33}}
\end{bmatrix} = \frac{\partial L}{\partial O}V^T
$$

## Gradient of $S$

To compute the gradient of $K$ and $Q$, we need to compute the gradient of $S$ first.

Remember that 

$$
P = \text{softmax}(S) = \begin{bmatrix}
\frac{exp(s_{11})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} & \frac{exp(s_{12})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} & \frac{exp(s_{13})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} \\
\frac{exp(s_{21})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} & \frac{exp(s_{22})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} & \frac{exp(s_{23})}{exp(s_{21}) + exp(s_{22}) + exp(s_{23})} \\
\frac{exp(s_{31})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})} & \frac{exp(s_{32})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})} & \frac{exp(s_{33})}{exp(s_{31}) + exp(s_{32}) + exp(s_{33})}
\end{bmatrix} = \begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{bmatrix}
$$

So for example $s_{11}$ appears in the first row of $P$, so

$$
\frac{\partial L}{\partial s_{11}} = \frac{\partial L}{\partial p_{11}}\frac{\partial p_{11}}{\partial s_{11}} + \frac{\partial L}{\partial p_{12}}\frac{\partial p_{12}}{\partial s_{11}} + \frac{\partial L}{\partial p_{13}}\frac{\partial p_{13}}{\partial s_{11}}
$$

Since $p_{11} = \frac{exp(s_{11})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})}$, 

$$
\frac{\partial p_{11}}{\partial s_{11}} = \frac{exp(s_{11})(exp(s_{11}) + exp(s_{12}) + exp(s_{13})) - exp(s_{11})exp(s_{11})}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = \frac{exp(s_{11})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})} - \frac{exp(s_{11})^2}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = p_{11} - p_{11}^2
$$

Since $p_{12} = \frac{exp(s_{12})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})}$,

$$
\frac{\partial p_{12}}{\partial s_{11}} = \frac{0 * (exp(s_{11}) + exp(s_{12}) + exp(s_{13})) - exp(s_{12})exp(s_{11})}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = - \frac{exp(s_{12})exp(s_{11})}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = -p_{11}p_{12}
$$

Since $p_{13} = \frac{exp(s_{13})}{exp(s_{11}) + exp(s_{12}) + exp(s_{13})}$,

$$
\frac{\partial p_{13}}{\partial s_{11}} = \frac{0 * (exp(s_{11}) + exp(s_{12}) + exp(s_{13})) - exp(s_{13})exp(s_{11})}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = - \frac{exp(s_{13})exp(s_{11})}{(exp(s_{11}) + exp(s_{12}) + exp(s_{13}))^2} = -p_{11}p_{13}
$$

So 

$$
\frac{\partial L}{\partial s_{11}} = \frac{\partial L}{\partial p_{11}}(p_{11} - p_{11}^2) + \frac{\partial L}{\partial p_{12}}(-p_{11}p_{12}) + \frac{\partial L}{\partial p_{13}}(-p_{11}p_{13})
$$

And similarly we could derive that

$$
\frac{\partial L}{\partial s_{12}} = \frac{\partial L}{\partial p_{11}}( - p_{11}p_{12}) + \frac{\partial L}{\partial p_{12}}(p_{12} -p_{12}^2) + \frac{\partial L}{\partial p_{13}}(-p_{12}p_{13})
$$

$$
\frac{\partial L}{\partial s_{13}} = \frac{\partial L}{\partial p_{11}}( - p_{11}p_{13}) + \frac{\partial L}{\partial p_{12}}(-p_{12}p_{13}) + \frac{\partial L}{\partial p_{13}}(p_{13} -p_{13}^2)
$$


Let's use $\frac{\partial L}{\partial S_{1}} = (\frac{\partial L}{\partial s_{11}}, \frac{\partial L}{\partial s_{12}}, \frac{\partial L}{\partial s_{13}})$, and $\frac{\partial L}{\partial P_{1}} = (\frac{\partial L}{\partial p{11}}, \frac{\partial L}{\partial p_{12}}, \frac{\partial L}{\partial p_{13}})$,  then we have

$$
\frac{\partial L}{\partial S_{1}} = \frac{\partial L}{\partial P_{1}} \begin{bmatrix}
p_{11} * (1-p_{11}) & -p_{11}p_{12} & -p_{11}p_{13} \\
-p_{11}p_{12} & p_{22} * (1-p_{22}) & -p_{12}p_{13} \\
-p_{11}p_{13}  & -p_{12}p_{13}  & p_{33} * (1-p_{33})
\end{bmatrix}
$$

Let $P_1 = (p_{11}, p_{12}, p_{13})$, then we have

$$
\begin{bmatrix}
p_{11} * (1-p_{11}) & -p_{11}p_{12} & -p_{11}p_{13} \\
-p_{11}p_{12} & p_{22} * (1-p_{22}) & -p_{12}p_{13} \\
-p_{11}p_{13}  & -p_{12}p_{13}  & p_{33} * (1-p_{33})
\end{bmatrix} = \begin{bmatrix}
p_{11} & 0 & 0 \\
0 & p_{12} & 0 \\
0 & 0 & p_{13}
\end{bmatrix} - P_1^T P_1
$$

So

$$
\frac{\partial L}{\partial S_{1}} = \frac{\partial L}{\partial P_{1}} \circ P_1 - (\frac{\partial L}{\partial P_{1}} P_1^T) P_1
$$

where $\circ$ is the element-wise product. And from the last section for $\frac{\partial L}{\partial P_{1}} = \frac{\partial L}{\partial O_{1}}V^T$ where $\frac{\partial L}{\partial O_{1}} = (\frac{\partial L}{\partial o_{11}}, \frac{\partial L}{\partial o_{12}}, \frac{\partial L}{\partial o_{13}})$. So we have

$$
\begin{align*}
\frac{\partial L}{\partial S_{1}} 
&= \frac{\partial L}{\partial P_{1}} \circ P_1 - (\frac{\partial L}{\partial O_{1}}V^T P_1^T) P_1 \\
&= \frac{\partial L}{\partial P_{1}} \circ P_1 - (\frac{\partial L}{\partial O_{1}}(P_1 V)^T) P_1 \\
&= \frac{\partial L}{\partial P_{1}} \circ P_1 - (\frac{\partial L}{\partial O_{1}}O_1^T) P_1 \\
&= \frac{\partial L}{\partial P_{1}} \circ P_1 - ROW\_SUM(\frac{\partial L}{\partial O_{1}} \circ O_1) P_1
\end{align*}
$$

So extending this to all rows, we have


$$
\begin{align*}
\frac{\partial L}{\partial S}
&= \frac{\partial L}{\partial P} \circ P - ROW\_SUM(\frac{\partial L}{\partial O} \circ O) \circ P
\end{align*}
$$


## Gradient of $Q$ and $K$

Since $S = QK^T$, we have

$$
\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S}K
$$

$$
\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial S})^T Q
$$

The derivation is similar to the gradient of $V$ and $P$.




























# References


\[1\]: @misc{dao2022flashattentionfastmemoryefficientexact,
      title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness}, 
      author={Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher Ré},
      year={2022},
      eprint={2205.14135},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2205.14135}, 
}

\[2\]: @misc{dao2023flashattention2fasterattentionbetter,
      title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning}, 
      author={Tri Dao},
      year={2023},
      eprint={2307.08691},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2307.08691}, 
}

\[3\]: @misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}


