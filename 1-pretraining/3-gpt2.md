
# GPT2

- **Generative Pre-trained Transformer**
    - decoder: only use a decoder, not an encoder
    - unidirectional: only use the left context
    - 1.5B parameter, trained on Reddit, a social media platform, which received at least 3 karma.
    - condition not only on the input but also on the task to be performed $p(output|input, task)$
    - GPT model size
        - GPT-2 : 1.5B
        - GPT-3 : 175B
        - GPT-4 : 10T
    - GPT variations
        - Distil GPT (A lighter version of GPT)
        - OpenAI Codex (for coding)

![Untitled](GPT2%20f19e13e386234edc8c4ebfdc491c9eff/Untitled%203.png)

- **Auto-regressive**
    - After each token is produced, that token is added to the sequence of inputs. And that new sequence becomes the input to the model in its next step.

- **decoder**
    - The GPT-2 is built using transformer `decoder` blocks. GPT2, like traditional language models, outputs one token at a time. 

Below are the encoder & decoder blocks for reference:

![Untitled](GPT2%20f19e13e386234edc8c4ebfdc491c9eff/Untitled.png)

![Untitled](GPT2%20f19e13e386234edc8c4ebfdc491c9eff/Untitled%201.png)

- **masked self-attention**
    - One key difference is that it **masks future tokens** – not by changing the word to [mask] like BERT, but by interfering in the self-attention calculation blocking information from tokens that are to the right of the position being calculated.
    - It’s important that the distinction between self-attention (what BERT uses) and masked self-attention (what GPT-2 uses) is clear. A normal self-attention block allows a position to peak at tokens to its right. Masked self-attention prevents that from happening.

![Untitled](GPT2%20f19e13e386234edc8c4ebfdc491c9eff/Untitled%202.png)


## Reference
- [Language models are unsupervised multitask learners](https://scholar.google.com/scholar?q=Language+Models+are+Unsupervised+Multitask+Learners&hl=en&as_sdt=0&as_vis=1&oi=scholart)
- [illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2/)
- [GPT model explained](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)
- [AndrejKarpathy GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)