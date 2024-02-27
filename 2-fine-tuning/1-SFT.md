
# SFT

**Instruct Tuning**

- Input : Instruction
- Output : the desired result by LLM

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled.png)

Stanford Alpaca : **high-quality instruction-following data**

- the [self-instruct](https://arxiv.org/abs/2212.10560) paper suggests using an existing strong language model to automatically generate instruction data.

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%201.png)

**FLAN-T5  :** an enhanced version of T5 that has been finetuned in a mixture of tasks.

Use instruction with `examples` and `chain-of-thought`

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%202.png)

# SPIN: Self-Play Fine-Tuning

- Input: LLM, **Supervised Dataset**, regularization parameter: λ
- Output: A fine tuned LLM which output more closely to the input dataset.

Self-Play example: AlphaGo Zero

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%203.png)

# References

- [Stanford Alphaca](https://www.notion.so/model-training-600ae0a5042a47ca81a7fb4be8b7e777?pvs=21)