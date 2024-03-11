# Instruct Tuning

## Format

- Input: (Instruction, Output) pair
    - Instruction: Human instructions
    - Output: Desired output
    - IT only captures **surface-level patterns** and styles rather than comprehending and learning the task

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled.png)

### Stanford Alpaca

**high-quality instruction-following data**

- the [self-instruct](https://arxiv.org/abs/2212.10560) paper suggests using an existing strong language model (GPT-3.5) to automatically generate instruction data.
- Alpaca, which is fine-tuned from Meta’s [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) 7B model. We train the Alpaca model on **52K instruction-following demonstrations** generated in the style of [self-instruct](https://arxiv.org/abs/2212.10560) using text-davinci-003. On the self-instruct evaluation set, Alpaca shows many behaviors similar to OpenAI’s text-davinci-003 (175B), but is also surprisingly small and easy/cheap to reproduce.

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%201.png)

### FLAN-T5

An enhanced version of T5 that has been finetuned in a mixture of tasks.

Use instruction with `examples` and `chain-of-thought` 

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%202.png)

# SPIN: Self-Play Fine-Tuning

- Input & output
    - Input: LLM, **Supervised Dataset**, regularization parameter: λ
    - Output: A fine-tuned LLM which outputs more closely to the input dataset.
- Self-Play: Eg: AlphaGo Zero
- The model **creates its own training data** from these self-conversations and learns to tell the difference between its own responses and those from humans.
- Steps
    1. Select a Supervised Fine-Tuned (SFT) LLM
    2. Generate New Synthetic Responses
    3. Create a Pairwise Dataset
    4. Train the LLM on the Pairwise Dataset
    5. Evaluate the Refined LLM

![Untitled](SFT%20&%20SPIN%20a3b2adba308749e3964877c20881cdcf/Untitled%203.png)

# References

- [Stanford Alphaca](https://www.notion.so/model-training-600ae0a5042a47ca81a7fb4be8b7e777?pvs=21)
- [SPIN paper](https://www.google.com/url?q=https://arxiv.org/pdf/2401.01335.pdf&sa=D&source=editors&ust=1710117901710216&usg=AOvVaw3B90xnpdLu9g6Mr81xwjPB)
- [Instruct Tuning](2212.10560)