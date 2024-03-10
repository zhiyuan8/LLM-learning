# LLM Learning Notes

## topic 0: Fundementals
- [Machine Learning basics](0-fundementals/1-rnn.md)
- [Transformers and Attention Mechanism](0-fundementals/2-attention.md)

## topic 1: LLM Pretraining
- [Google BERT](1-pretraining/1-bert.md)
- [Google T5](1-pretraining/2-t5.md)
- [Meta LLaMa2](1-pretraining/3-llama2.md)
- [Google PaLM2](1-pretraining/4-palm2.md)
- [Mistral-8x7B](1-pretraining/5-mistral.md)
- [Google Gemma](1-pretraining/6-gemma.md)

## topic 2: Fine-tuning
- SFT (instruction tuning)
- SPIN
- Prompt engineering / Tuning
- Prefix tuning
- LoRA & QLoRA


## topic 3 : Efficient Training and Inference
- Parallelization
- FlashAttention and Flash Attention 2
- Linear Attention: Lightening Attention 2
- Quantization
- Speculative Decoding
- Overview of Efficient Serving (if time permit)

## topic 4: Agent and Applications
- RAG
- Tool-using
- Open-source resources
  - Hugging face
  - Ray
  - langchain
  - Triton
  - Format LLM

## top 5: Reinfocement Learning
- PPO, DPO
- RLHF
- RLAiF

## top 6: on-device ML
- MobileLLM (Meta)
- AppAgent (Tencent) and MobileAgent (Alibaba)
- MLC LLM (Machine Learning Compilation for Large Language Models)
- Pytorch Mobile


Below is model comparison table:

To make the model comparison table fit better in GitHub Markdown and improve readability, we can split it into multiple smaller tables centered around specific themes or categories of information. This approach makes it easier to digest and allows for more focused comparison within specific areas of interest. Here's how you can reformat the table:

### Company and Time

| Model      | Company  | Time |
|------------|----------|------|
| BERT       | Google   | 2018 |
| T5         | Google   | 2019 |
| GPT-1      | OpenAI   | 2018 |
| GPT-2      | OpenAI   | 2019 |
| GPT-3      | OpenAI   | 2020 |
| GPT-4      | OpenAI   | 2023 |
| PALM       | Google   | 2023 |
| LLAMA      | Facebook | 2023 |
| LLAMA 2    |          | 2023 |

### Architecture, Model Size, and Objective

| Model      | Architecture       | Model Size           | Objective                                                  |
|------------|-------------------|----------------------|------------------------------------------------------------|
| BERT       | encoder            | 110M, 340M           | MLM, NSP                                                   |
| T5         | enc-dec            | 220M-11B             | Text-to-Text                                               |
| GPT-1      | casual-decoder     | 117M                 | CLM (Casual lang model)                                    |
| GPT-2      | casual-decoder     | 1.5B                 | CLM                                                        |
| GPT-3      | casual-decoder     | 175B                 | CLM                                                        |
| GPT-4      | MOE                | 1.76T?               | "Performance, alignment, Auxiliary Objective"             |
| PALM       | casual-decoder     | 8,62,540B            | CLM                                                        |
| LLAMA      | casual-decoder     | 7-65B                | CLM, auto-regressive                                       |
| LLAMA 2    | casual-decoder     | 70B                  |                                                            |

### Data, Preprocessing, and Tokenization

| Model      | Data                                                                 | Preprocessing | Tokenizer                                     |
|------------|----------------------------------------------------------------------|---------------|-----------------------------------------------|
| BERT       | BookCorpus, EnWiki                                                   | no            | wordpiece                                    |
| T5         | C4(en)                                                               | Task Prefix   | sentencepiece(wordpiece, 32000)              |
| GPT-1      | BookCorpus                                                           | no            | BPE                                          |
| GPT-2      | WebText(40GB)                                                        | task + {q}, {a}| BPE(50257)                                    |
| GPT-3      | Mix (Common Crawl, WebText2, Books1, Books2 and Wikipedia.)         | InContextLearning | BPE (variant, SentencePiece)           |
| GPT-4      |                                                                      |               | SentencePiece 256k                           |
| PALM       | public data                                                          |               | BPE (variant, SentencePiece)                 |
| LLAMA      | OpenSource                                                           |               | BPE (variant, SentencePiece, 32k)            |

### Technical Specifications

| Model      | PositionalEncoding         | Attention          | FFW+Activation+related                      | ContextLength | Layers | BatchSize |
|------------|----------------------------|--------------------|---------------------------------------------|---------------|--------|-----------|
| BERT       | absolute positional encoding, segment embedding,learnable | bidirectional | original,gelu                              | 512           | 12,24  | 32        |
| T5         | relative positional encoding| encoder: fully visible, decoder: casual attention mask. prefix LM | original        | 512     | 12      | 128       |
| GPT-1      | absolute positional encoding, learnable |                  | GLEU                                        | 512           | 12     | 64        |
| GPT-2      | absolute positional encoding|                    | GLEU, LayerNorm before decoder              | 768,1024, 1280, 1600 | 48   | 512       |
| GPT-3      | absolute positional encoding|                    | PreNorm                                    | 2048          | 96     | 32M       |
| GPT-4      |                            | MQA                |                                            |               | 120    | 16M       |
| PALM       | ROPE(variant)              | SparseAttention    | SwiGLU                                      | 4,096,819,218,432 | 32,64,118 |         |
| LLAMA      | ROPE(variant)              | GroupQueryAttention| PreNorm, SwiGLU, 2.7x(instead of 4)        | 2k            | 32-80  | 4M        |
| LLAMA 2    |                            |                    | PreNorm(RMSNorm), SwiGLU                    | 4k            | 32-80  | 4M        |