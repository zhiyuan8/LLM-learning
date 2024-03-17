# FlashAttention

Goal: **avoid reading and writing the attention matrix to and from HBM**

- making attention algorithms `IO-aware`
- implement FlashAttention in CUDA
- runs faster (up to 7.6x on GPT-2) and uses less memory (linear in sequence length)
- work for both training and inference

![Untitled](FlashAttention%20fabb39ee8db24521809dd5ee0a23fb75/Untitled.png)

## GPU knowledge

A100 GPU has 40-80GB of `high bandwidth memory` (HBM) with bandwidth 1.5-2.0TB/s and `192KB` of `on-chip SRAM` per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s

- **[Static random-access memory](https://en.wikipedia.org/wiki/Static_random-access_memory) (**SRAM)
- **[High Bandwidth Memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)** (HBM)
- **Compute bound**: matrix multiply with large inner dimension, and convolution with large number of channels.
- **Memory-bound**: elementwise (e.g., `activation`, `dropout`), and reduction (e.g., sum, `softmax`, `batch norm`, `layer norm`).

## A100 GPU knowledge

A100 GPU has `40-80GB` of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and `192KB` of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s

## How FlashAttention works

- `tiling` : compute attention by blocks. Softmax couples columns of K, so we decompose the large softmax with scaling.
    - Let N be the sequence length, d be the head dimension, and M be size of SRAM with $d < M < Nd$.
    - Standard attention requires $Θ(Nd + N^2)$ HBM accesses, while FLASHATTENTION requires $Θ(N^2d^2M^{-1})$ HBM accesses.

For typical values of 𝑑 (64-128) and **𝑀 (around 100KB)**, $d^2$ is many times smaller than 𝑀, and thus FlashAttention requires many times fewer HBM accesses than standard implementation

# References

- [Meet Notes](https://docs.google.com/document/d/1B0Xhyyqgz_fqjj7e7jD_eIK1ByygM48VJyL5J_zEByI/edit#heading=h.yv7qylxozth0)
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt engineering with the OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)