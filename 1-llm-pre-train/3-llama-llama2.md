
# LLaMA2

- Summary
    - LLaMA-2 with **7B, 13B, and 70B** parameters
    - context length 4k tokens
    - trained on 2 trillion tokens of data

![Untitled](llama%20&%20llama2%2045ff5994ec6b4b4da2b3f626f261c59a/Untitled.png)

- tokenization
    - bytepair encoding (BPE)
- training
    - standard transformer architecture
    - RMSNorm as pre-normalization
    - **SwiGLU** activation function
    - **RoPE**: Rotary Position Embeddings
    - **GQA**: grouped-query attention: allows the model to use different strategies for managing attention heads, potentially enhancing performance on certain tasks.
    - AdamW Optimizer
- fine-tuning
    - **Ghost Attention** (GAtt): The **loss of context** in multi-turn conversations has been recognized as a known issue.
    - **SFT**: tens of thousands of annotations are enough to achieve a high-quality result
    - **RLHF**: further *align* model behavior with human preferences and instruction following.
- **Reinforcement Learning with Human Feedback (RLHF)**
    - two **reward models** for Safety and Helpfulness
    - **Proximal Policy Optimization (PPO)** : uses the reward model as an estimate for the true reward function (human preference) and the pre-trained language model as the policy to optimize.
    - **Rejection Sampling fine-tuning** :   Sample K outputs from the model and select the best candidate with a reward, and use the selected outputs for a gradient update.
- llama2 [prompt template](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) :

```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```

## LLaMA2 Huggingface transformers
configuration_llama.py
- vocab_size: the number of unique tokens the model can understand and generate, = 30k for llama-7B
- num_attention_heads: the number of attention heads in the multi-head attention mechanism, = 32 for llama-7B
- num_key_value_heads: for grouped query attention
- max_position_embeddings: the maximum length of a sequence that the model can handle, = 2048 for llama-7B
- use_cache: whether to use the cache in the forward pass, = True for llama-7B
- rope_theta and rope_scaling: Parameters related to Rotary Position Embeddings (RoPE), which are experimental features for enhancing the model's handling of positional information.

modeling_llama.py
- **`LlamaPreTrainedModel`**: Base class for model-specific classes, handling weight initialization, and model checkpoint management.
- **`LlamaModel`**: Core Transformer with embeddings and Transformer decoder layers, versatile for various tasks.
- **`LlamaForCausalLM`**: Adds a linear layer to `LlamaModel` for next-token prediction in language modeling.
- **`LlamaForSequenceClassification`** & **`LlamaForQuestionAnswering`**: Add task-specific heads for classification and question answering.

Layer
- **`LlamaDecoderLayer`**: a critical component of the LLaMA model architecture, representing a single layer within the Transformer decoder. Each decoder layer is designed to process and transform the input sequence through a series of operations, including self-attention and position-wise feed-forward networks.

Embedding
- **`LlamaRotaryEmbedding`**: Incorporates positional information into attention, enhancing task performance sensitive to token order.
- **`LlamaLinearScalingRotaryEmbedding`**: Extends `LlamaRotaryEmbedding` with a linear scaling factor for dynamic sinusoidal embedding frequency adjustment.
- **`LlamaDynamicNTKScalingRotaryEmbedding`**: Utilizes NTK theory for dynamic scaling, optimizing embeddings across variable sequence lengths.

Attention
- **LlamaFlashAttention2**: The LlamaFlashAttention2 class is an advanced implementation of the attention mechanism, designed to leverage the Flash Attention algorithm for faster and more memory-efficient computation.
- **LlamaSdpaAttention**: an implementation of the Scaled Dot-Product Attention (SDPA) within the LLaMA model architecture.


# LLaMA
- **SwiGLU activation function**
- **rotary positional embeddings (RoPE)**

# Reference
- [Understand llama2 architecture](https://medium.com/towards-generative-ai/understanding-llama-2-architecture-its-ginormous-impact-on-genai-e278cb81bd5c)
- [Llama 2: A Comprehensive Guide](https://www.simform.com/blog/llama-2-comprehensive-guide/)
- [llam2 huggingface](https://huggingface.co/blog/llama2)
- [llama2 intro](https://huggingface.co/blog/llama2)
- [Meta llama2 code](https://github.com/facebookresearch/llama)
- [huggingface transformers with llama2](https://github.com/huggingface/transformers/tree/da20209dbc26a6a870a6e7be87faa657b571b7bc/src/transformers/models/llama)
- [huggingface text generation inference](https://github.com/huggingface/text-generation-inference)
- [model card](https://huggingface.co/meta-llama/Llama-2-7b)
- [model quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)