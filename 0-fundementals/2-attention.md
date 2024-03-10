
# Attention Mechanism

## **Problems with RNNs**

1. **Slow Computation:** Processing is sequential, leading to longer training times.
2. **Vanishing Gradient:** Difficulty in capturing long-term dependencies in sequences.
3. **Memory Limitations:** Struggles with retaining information from long input sequences.

## **How Attention Improves Model Performance**

1. **All Hidden States Are Used:** Unlike encoder-decoder architectures that only pass the final hidden state, the attention mechanism leverages all hidden states, allowing the model to access the entire sequence history.
2. **Dynamic Focus:** The model dynamically focuses on different parts of the input sequence, which is particularly useful for tasks like machine translation where the relevance of input parts can vary greatly.

## **Types of Attention**

### Q-K-V

- **Query (Q):** Represents the current element or word we're focusing on.
- **Key (K):** Represents all possible elements or words we can attend to.
- **Value (V):** Represents the actual content of the elements or words we can attend to.

1- **Self attention** Q = K = V = Our source sentence(English)

2- **Decoder Self attention** Q = K = V = Our target sentence(German)

3- **Decoder-Encoder attention** Q = Our target sentence(German) K = V = Our source sentence(English)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- Q (Queries) dimension: $n \times d_k$
- K (Keys) dimension: $m \times d_k$
- V (Values) dimension: $m \times d_v$
- n is the number of tokens (e.g., words) in the sequence
- m is the dimension of embedding vector

## Additive Mechanism

When query and key have difference length

Additive attention computes the compatibility function using a feed-forward network with dk
a single hidden layer.

## Dot-Product Mechanism

![Untitled](Transformers%20&%20Attention%2036e31c8a100e414ab5ac29c33bb9f95a/Untitled.png)

### **1. Self-Attention**

Enable the model to weight the importance of word-pairs. For below sentence, self-attention finds “it” is closed to “The animal”.

`The animal didn't cross the street because it was too tired`

### **2. Encoder-Decoder Attention**

Allows the decoder to focus on relevant parts of the input sequence, enhancing tasks like translation where the target depends on specific input parts.

### **3. Single-Head vs. Multi-Head Attention**

- **Single-Head Attention:** Processes queries, keys, and values in one set, producing a single set of outputs (Z).
- **Multi-Head Attention:** Utilizes multiple sets of Query, Key, and Value weight matrices $W^Q, W^K, W^V$ to process the input in parallel heads, **use different attention functions**, allowing the model to capture different aspects of the information.

![Untitled](Transformers%20&%20Attention%2036e31c8a100e414ab5ac29c33bb9f95a/Untitled%201.png)

### **4. Masked Attention**

For Decoder, ensures that the prediction for a certain position is only dependent on the known outputs at positions before it.

# Transformers

The Transformer architecture is composed of an encoder stack and a decoder stack (N=6)

![Untitled](Transformers%20&%20Attention%2036e31c8a100e414ab5ac29c33bb9f95a/Untitled%202.png)

### 1. Embedding

**Positional embedding :** position + token embeddings

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
  PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

### **2. encoder block**

An encoder receives a list of vectors as input. It processes this list by passing these vectors into a ‘self-attention’ layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.

![Untitled](Transformers%20&%20Attention%2036e31c8a100e414ab5ac29c33bb9f95a/Untitled%203.png)

**Position-wise Feed-Forward Networks**: It is just a `MLP` with one hidden layer. Fully connected layers used in both the encoder and decoder layers to learn complex relationships between the input embeddings and their context. The FFN typically consists of two linear transformations with a ReLU activation in between.

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

where `\max(0, xW_1 + b_1)` is just ReLU.

**Layer Normalization and Residual Connection**: After each of the two main components (self-attention and FFN), a residual connection followed by layer normalization is applied. This can be formulated as:

$$
  \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

$$
 \text{LayerNorm}(x) = \gamma \left( \frac{x - \mu}{\sigma} \right) + \beta
$$

![Untitled](Transformers%20&%20Attention%2036e31c8a100e414ab5ac29c33bb9f95a/Untitled%204.png)

Why not using `Batch Norm` ?

BatchNorm is aimed at stabilizing the numerical environment of the network by normalizing layer inputs, while attention mechanisms aim to model dependencies and relationships within the data. 

### **3. decoder block**

Each decoder block in the Transformer model includes three main sub-layers:

- **Masked Self-Attention**: Similar to the encoder, but **with masking applied to prevent positions** from attending to subsequent positions. This ensures that the predictions for position \(i\) can only depend on the known outputs at positions less than \(i\).
- **Encoder-Decoder Attention**: This allows each position in the decoder to attend to all positions in the encoder. This is where the decoder processes the encoder's output. Queries come from the previous decoder layer, while keys and values come from the encoder's output.
- **Position-wise Feed-Forward Networks**: Identical to those in the encoder block.

The same residual connection and layer normalization strategy are applied after each sub-layer in the decoder as well.

# References

[Transformers Explained Visually](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

[Transformers Explained](https://jalammar.github.io/illustrated-transformer)