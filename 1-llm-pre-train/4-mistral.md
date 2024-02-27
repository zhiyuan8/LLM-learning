# Mixtral 8x7B: Sparse Mixture of Expert
- Mixtral 8x7B model (mistral small) = Mistral 7B + MoE Architecture
- Mixtral 47B parameters, but only uses 13B active parameters during inference. Mistral Medium: state of arts SMoE, but not open source

# MoE

$$
\sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)
$$

G(x)i denotes the n-dimensional output of the gating network for the i-th expert, and Ei(x) is the output of the i-th expert network.

A simple and performant one is implemented by taking the softmax over the Top-K logits of a linear layer

$$
G(x) := \text{Softmax}(\text{TopK}(x \cdot W_g))
$$

For Mixtral, the same SwiGLU architecture is used as the expert function Ei(x) and set K = 2. This means each token is routed to two SwiGLU sub-blocks with different sets of weights.

$$
y = \sum_{i=0}^{n-1} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x)
$$

**Why not 1000 x 7b ?**

More experts lead to improved sample efficiency and faster speedup, but these are diminishing gains (especially after 256 or 512), and more VRAM will be needed for inference. The properties studied in Switch Transformers at large scale were consistent at small scale, even with 2, 4, or 8 experts per layer.

**Challenges of SMoEs**

- Hard to fine tune and easy to overfit
- VRAM

# **Mixtral 8*7B**

Mixtral 8x7B is a Sparse Mixture of Experts (SMoE) language model trained with multilingual data using a context size of 32k tokens.

### Sparse Mixture of Experts (SMoE)

Each layer is composed of 8 feedforward blocks (i.e. experts)

At every layer, for every token, a router network chooses two of these groups (the “experts”) to process the token and combine their output additively.

Even though each token only sees two experts, the selected experts can be different at each timestep.

Each token has access to 47B parameters, but only uses 13B active parameters during inference.

![Untitled](Mixtral%208x7B%20Sparse%20Mixture%20of%20Expert%2091b6545d9f8c49718cd6940d24429a18/Untitled.png)

# **Mixtral 7B**

### Mixture of Experts (MoE)

Mixture of experts (MoE) is a machine learning technique where multiple expert networks (learners) are used to divide a problem space into homogeneous regions.

### Sliding Window Attention (SWA)

effectively handle sequences of arbitrary length with a reduced inference cost.

![Untitled](Mixtral%208x7B%20Sparse%20Mixture%20of%20Expert%2091b6545d9f8c49718cd6940d24429a18/Untitled%201.png)

By employing a window size of W = 4096, SWA theoretically achieves an attention span of approximately 131K tokens.

**Rolling Buffer Cache**

![Untitled](Mixtral%208x7B%20Sparse%20Mixture%20of%20Expert%2091b6545d9f8c49718cd6940d24429a18/Untitled%202.png)

A Rolling Buffer Cache, employs a fixed attention span to limit cache size. The cache is of fixed size W, and it stores keys and values for timestep i at position i mod W in the cache. When i exceeds W, earlier values are overwritten, halting cache size growth. For instance, with W = 3, on a 32k-token sequence, cache memory usage is reduced by 8x without compromising model quality.

**Pre-fill and chunking**

![Untitled](Mixtral%208x7B%20Sparse%20Mixture%20of%20Expert%2091b6545d9f8c49718cd6940d24429a18/Untitled%203.png)

In sequence generation, tokens are predicted sequentially based on prior context. To optimize efficiency, a `(k, v) cache` is pre-filled with the known prompt. If the prompt is very long, it is chunked into smaller segments using a chosen window size. Each chunk is used to pre-fill the cache. This approach involves computing attention both within the cache and over the current chunk, Thus aiding in more effective sequence generation.

**Mistral 7B - Instruct**

model fine-tuned to follow instructions on instruction datasets publicly available on the Hugging Face repository

# References

- [MOE](https://medium.com/@adamlouly/mixtral-8x7b-redefining-ai-with-mixture-of-experts-moe-technology-20b8f04c18a9) explained
- [Mixtral 8*7B paper explained](https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745)
- [Mixtral 7B paper explained](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580)
- [huggingface](https://www.notion.so/Paper-Reading-98ecff9f57374deba862c34c6280ac83?pvs=21)