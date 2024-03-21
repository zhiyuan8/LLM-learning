# Must Learn

- [Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/en/index)
- [TRL - Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [Datasets](https://huggingface.co/docs/datasets/index)

# 1. Transformers

## basics

- `pipeline`
- `AutoTokenizer`: the tokenizer returns a dictionary containing:
    - **[input_ids](https://huggingface.co/docs/transformers/en/glossary#input-ids)** are the indices corresponding to each token in the sentence.
    - **[attention_mask](https://huggingface.co/docs/transformers/en/glossary#attention-mask)** indicates whether a token should be attended to or not.
    - **[token_type_ids](https://huggingface.co/docs/transformers/en/glossary#token-type-ids)** identifies which sequence a token belongs to when there is more than one sequence.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

"""
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

# the tokenizer added two special tokens - CLS and SEP (classifier and separator) - to the sentence. 
tokenizer.decode(encoded_input["input_ids"])
"""
'[CLS] in a hole in the ground there lived a hobbit. [SEP]'
"""
```

- batch processing
    - `padding`
    - `truncation`

```jsx
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True,\
truncation=True, return_tensors="pt", max_length=512)
```

## Trainer

### Trainer implementation

- The **[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)** supports `distributed training` and `mixed precision`
- [https://github.com/zhiyuan8/LLM-learning/blob/main/huggingface/fine_tune_with_trainer/test_trainer.py](https://github.com/zhiyuan8/LLM-learning/blob/main/huggingface/fine_tune_with_trainer/test_trainer.py)

### Pytorch implementation

- [https://github.com/zhiyuan8/LLM-learning/blob/main/huggingface/fine_tune_with_trainer/test_pytorch_training.py](https://github.com/zhiyuan8/LLM-learning/blob/main/huggingface/fine_tune_with_trainer/test_pytorch_training.py)

## Common Pitfalls

1. The generated output is too short/long, set `max_new_tokens`
2. Use `do_sample` to add creativity
3. Since LLMs are not trained to continue from pad tokens, your input needs to be left-padded.

```jsx
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
```

## Transformer References

[https://huggingface.co/docs/transformers/notebooks](https://huggingface.co/docs/transformers/notebooks)

# 2. PEFT

### LoraConfig

### `lora_dropout`

- **Definition**: This parameter specifies the dropout probability for the LoRA layers. Dropout is a regularization technique used to prevent overfitting by randomly dropping units (along with their connections) from the neural network during training.
- **Mathematical Context**: If we denote the dropout operation by $\mathcal{D}(\cdot)$, then for any input x to a LoRA layer, the output after applying dropout is $\mathcal{D}(x; p)$, where \(p\) is the `lora_dropout` probability. This means each element of x has a probability p of being set to zero during the forward pass in training.

### `r` (Lora attention dimension, or "rank")

- **Definition**: The rank r represents the dimensionality of the low-rank approximation in the LoRA adaptation mechanism. It's a critical hyperparameter that controls the model's capacity to adapt to new tasks, balancing between flexibility and parameter efficiency.
- **Mathematical Context**: In the low-rank update $W + \alpha AB$, the matrices A and B have dimensions $d \times r$ and $r \times d$, respectively. The rank r not only affects the computational complexity but also the capacity of LoRA layers to model the adaptations. **A higher r means more parameters and greater adaptability**, at the cost of increased computational resources and potential overfitting.

### `target_modules`: "q_proj", "k_proj", "v_proj", "out_proj"

- **Definition**: This parameter specifies the modules within a transformer model to which LoRA is applied. For models based on the attention mechanism, such as those following the architecture of transformers, the modules typically targeted for adaptation are the projections for queries q_proj, keys k_proj, values v_proj, and outputs out_proj.
- **Mathematical Context**: In the context of a transformer, the attention mechanism can be summarized as computing attention scores based on query Q, key K, and value V matrices, followed by an output projection. LoRA selectively applies low-rank adaptations to the weight matrices of these projections, enabling fine-tuning and task-specific adaptation without altering the entire model structure.

### `bias`

- **Definition**: The `bias` parameter controls how biases are handled within LoRA-adapted layers. Options typically include `none`, `all`, or `lora_only`, determining whether biases are updated during training and if so, which ones.
- **Mathematical Context**: Consider a layer's output computed as $y = Wx + b$, where b is the bias vector. If `bias` is set to 'all', both W (along with its LoRA update) and b are trainable. If 'lora_only', only the biases in the LoRA-specific parts (if any) are trainable. 'None' means biases are not updated during training, keeping the model's original bias parameters fixed.

### `task_type`

- **Definition**: Although not explicitly mentioned in your initial query and not a parameter of the LoRA configuration, `task_type` could refer to the type of task for which the LoRA-adapted model is being fine-tuned (e.g., classification, regression, etc.). This contextual information could guide the selection and tuning of other parameters, including those discussed above.
- **Mathematical Context**: The `task_type` might not have a direct mathematical representation in the context of LoRA parameters, but it influences the overall design and fine-tuning strategy. For instance, different tasks may require different levels of adaptability and thus different settings for parameters like `lora_alpha` and `r`.

Example:

```jsx
peft_config = LoraConfig(
     r=8,
     lora_alpha=16,
     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
     lora_dropout=0.05,
     bias="none",
     task_type="CAUSAL_LM"
)
```

# 3. TRL

# 4. Datasets

**performance optimization tools to train and run models**

[https://huggingface.co/docs/optimum/index](https://huggingface.co/docs/optimum/index)

## 5. Accelerate

 **[Accelerate](https://huggingface.co/docs/accelerate)** library to help users easily train a Transformers model on any type of distributed setup, whether it is multiple GPU’s on one machine or multiple GPU’s across several machines.

**Deploy model for inference**

[https://huggingface.co/docs/inference-endpoints/pricing](https://huggingface.co/docs/inference-endpoints/pricing)

[https://huggingface.co/docs/inference-endpoints/index](https://huggingface.co/docs/inference-endpoints/index)

**UI**
[https://huggingface.co/docs/transformers.js/index](https://huggingface.co/docs/transformers.js/index)