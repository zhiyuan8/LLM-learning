
# BERT & GPT2 & T5

Owner: Zhiyuan Li
Tags: Guides and Processes

## Encoder vs Decoder vs Encoder-Decoder

**Encoder only** : highly efficient for tasks that require understanding the input without needing to generate new text, such as Name Entity Recognition (NER) and text classification. Examples : `BERT`

**Decoder only** : Unlike encoder models, they are designed to **`predict next token`** in a sequence given the previous tokens, making them well-suited for tasks like **text generation**. This architecture allows for `auto-regressive generation`, where the model predicts the next token based on all preceding tokens. Examples : `GPT` , `llama`

**Encoder-decoder models** : computationally intensive. `BART`, `T5` 

![Untitled](BERT%20&%20GPT2%20&%20T5%20570e753c77cf410eb92624478547513d/Untitled.png)

# BERT

- **Bidirectional Encoder Representations from Transformers**
    - bidierectional : use both left and right context
    - encoder : only use encoder, not decoder
    - representations : use representations of all layers, not just the last layer
    - transformers : use transformers, not RNNs or CNNs
- BERT pre-training :
    - object 1 : **Masked Language Modeling (MLM)**
        - randomly mask 15% of the words in each sequence
        - the model is trained to predict these masked tokens based only on their context.
        - Make model understand language context and relationships between words.
    - object 2 : **next sentence prediction**
        - the model is given pairs of sentences and must predict whether the second sentence follows the first in the original document.
        - This task helps BERT understand relationships between sentences.
    
    ![Untitled](BERT%20&%20GPT2%20&%20T5%20570e753c77cf410eb92624478547513d/Untitled%201.png)
    
- BERT size

![Untitled](BERT%20&%20GPT2%20&%20T5%20570e753c77cf410eb92624478547513d/Untitled%202.png)


# Reference
- [BERT](https://jalammar.github.io/illustrated-bert/)
- [BERT huggingface](https://huggingface.co/blog/bert-101)