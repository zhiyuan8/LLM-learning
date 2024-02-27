# LLM Learning Notes

## topic 0: Fundementals
- Reinforcement Learning (PPO, DPO)
- Transformers and Attention Mechanism
- GPT2

## topic 1: LLM Pretraining
- BERT
- T5
- LLaMa2
- PaLM
- Mistral-8x7B


## topic 2: Fine-tuning
- SFT (instruction tuning)
- RLHF (PPO and DPO basics)
- RLAiF
- SPIN
- Prompt engineering / Tuning
- Prefix tuning


## topic 3 : Efficient Training and Inference
- LoRA & QLoRA
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

Below is model comparison table:
|              | BERT    | T5          | GPT 1     | GPT2    | GPT3    | GPT4    | PALM       | LLAMA      | LLAMA 2      |
|--------------|---------|-------------|-----------|---------|---------|---------|------------|------------|--------------|
| Company      | Google  | Google      | OpenAI    | OpenAI  | OpenAI  |         | Google     | Facebook   |              |
| Time         | 2018    | 2019        | 2018      | 2019    | 2020    | 2023    | 2023       | 2023       | 2023         |
| Architecture | encoder | enc-dec     | casual-decoder | casual-decoder | casual-decoder | MOE | casual-decoder | casual-decoder | casual-decoder |
| ModelSize    | 110M, 340M | 220M-11B | 117M     | 1.5B   | 175B   | 1.76T?  | 8,62,540B  | 7-65B      | 70B           |
| Objective    | MLM, NSP | Text-to-Text | CLM (Casual lang model) | CLM | CLM | "Performance, alignment, Auxiliary Objective" | CLM | CLM, auto-regressive |
| Data         | BookCorpus, EnWiki | C4(en) | BookCorpus | WebText(40GB) | Mix (Common Crawl, WebText2, Books1, Books2 and Wikipedia.) | | public data | OpenSource |
| Preprocessing| no      | Task Prefix | no        | task + {q}, {a} | InContextLearning | | | |
| Tokenizer    | wordpiece | sentencepiece(wordpiece, 32000) | BPE | BPE(50257) | BPE (variant, SentencePiece) | SentencePiece 256k | BPE (variant, SentencePiece) | BPE (variant, SentencePiece, 32k) |
| PositionalEncoding | absolute positional encoding, segment embedding,learnable | relative positional encoding | absolute positional encoding, learnable positional encoding | absolute positional encoding | absolute positional encoding | | ROPE(variant) | ROPE(variant) |
| Attention    | bidirectional | encoder: fully visible, decoder: casual attention mask. prefix LM | | | | MQA | SparseAttention | GroupQueryAttention |
| FFW+Activation+related | original,gelu | original | GLEU | GLEU, LayerNorm before decoder | PreNorm | | SwiGLU | PreNorm, SwiGLU, 2.7x(instead of 4) | PreNorm(RMSNorm), SwiGLU |
| ContextLength| 512     | 512         | 512       | 768,1024, 1280, 1600 | 2048    | | 4,096,819,218,432 | 2k          | 4k            |
| Layers       | 12,24   | 12          | 12        | 48      | 96      | 120     | 32,64,118  | 32-80       | 32-80         |
| BatchSize    | 32      | 128         | 64        | 512     | 32M     | 16M     | | 4M        | 4M            |
| Evaluation   | Glue    | cross-entropy loss | | | | zero-one-few shot | | |
| Training technique | | | pre-train + fine tune | one-train | | | RLHF | |
| Optimizer    | | AdaFactor | | | | | AdamW | AdamW |

