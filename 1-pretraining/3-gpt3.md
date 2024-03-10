# GPT-3

- autoregressive language model with 175 billion parameters, trained on 300 billion tokens.
- in-context learning / `few-shot learning`
    - For tasks, it uses few-shot learning ONLY, NO gradient updates or fine-tuning
- `meta-learning`
    - the model develops a broad set of skills and pattern recognition abilities at training time, and then uses those abilities at inference time to rapidly adapt to or recognize the desired task
- Limitation
    - Our current objective **weights every token equally** and lacks a notion of what is most important to predict and what is less important.
- Train process

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled.png)

- Inference process

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled%201.png)

### **Model Architecture**:

GPT-3 follows the **transformer-based architecture**, **decoder-only**, and retains the **masked self-attention mechanism**.

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled%202.png)

### **Pre-training**

train for 300 billion tokens, some datasets are seen up to 3.4 times during training while other datasets are seen less than once.

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled%203.png)

To train all versions of GPT-3, we use Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$, and $\epsilon = 10^{-8}$, we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. All models use a weight decay of 0.1 to provide a small amount of regularization.

During training we always train on sequences of the full $n_{ctx} = 2048$ token context window, packing multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency. Sequences with multiple documents are not masked in any special way but instead, documents within a sequence are delimited with a special end-of-text token, giving the language model the information necessary to infer that the context separated by the end-of-text token is unrelated. This allows for efficient training without the need for any special sequence-specific masking.

## Scaling Law

Performance (measured in terms of cross-entropy validation loss) follows a power-law trend with the amount of computing used for training.

```markdown
**The LM’s test loss varies as a power law with respect to the number of model parameters**
```

In math, it is described as 

$$
y = ax^p
$$

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled%204.png)

While zero-shot performance improves steadily with model size, few-shot performance increases more rapidly, demonstrating that larger models are more proficient at in-context learning.

![Untitled](GPT3%20cf1708e67bc54046b45c31313035a52e/Untitled%205.png)

### **References**:

- [Language Models are Few-Shot Learners](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&as_vis=1&q=Language+Models+are+Few-Shot+Learners&btnG=)
- [AndrejKarpathy GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)