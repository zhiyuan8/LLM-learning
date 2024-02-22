# Recurrent Neural Networks (RNNs)

In essence, at each time step , an RNN takes two inputs:

1. **An input vector** $x_t$ , which could be, for example, the embedding of a word in a sentence.
2. **A hidden state** $h_{t-1}$, which is the network's memory of the previous inputs.

The RNN then produces an output, $y_t$, and updates its hidden state $h_t$, which is passed on to the next time step:

$$
h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
$$

$$
y_t = g(W_{hy}h_t + b_y)
$$

where f and g are non-linear activation functions, $W_{hh}$, $W_{xh}$, and $W_{hy}$ are weight matrices, and $b_h$ and $b_y$  are bias vectors.

![Untitled](Seq2Seq%20&%20RNN%20927b8057e1e74a8ca8c608feb367c468/Untitled.png)

## Sequence-to-Sequence (Seq2Seq) with RNN

Seq2Seq models are a specific application of RNNs for tasks where the input and output are both sequences that may have different lengths.A Seq2Seq model consists of two parts:

1. **Encoder**: Processes the input sequence and compresses the information into a `context vector` (or `state vector`), which is the `final hidden state` of the encoder RNN.
2. **Decoder**: Starts with the context vector to generate the output sequence.

[seq2seq demo](Seq2Seq%20&%20RNN%20927b8057e1e74a8ca8c608feb367c468/seq2seq_6.mp4)

## Seq2Seq with Attention Mechanism

The attention mechanism was introduced to overcome the limitations of the traditional Seq2Seq.

1. **All hidden states from the encoder are passed to the decoder**. Instead of using just the last hidden state, the decoder has access to all the hidden states from the encoder
2. **The decoder computes attention scores for all encoder hidden states** to decide which states are most relevant at each decoding step:
    1. The decoder examines all encoder hidden states, $\{h_1, h_2, ..., h_N\}$, where each state is associated with a specific word in the input sentence.
    2. **Scoring Function :** To determine the relevance of each encoder hidden state $h_i$ to the decoder's hidden state $s_t$, a scoring function is applied and it reflects the importance of the input word for generating the current output word. A common choice is the dot product, although other functions like additive or concat-based functions can also be used:  
    
    **Dot product**: $\text{score}(s_t, h_i) = s_t^\top h_i$  
    
    **Additive/Concat**: $\text{score}(s_t, h_i) = v^\top \tanh(W_1 s_t + W_2 h_i)$
    
    Here, $W_1,W_2,v$ are learnable parameters of the scoring function.
    
    c.  **Softmax Normalization and Weighting**. Multiply each hidden state by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores
    
[attention process](Seq2Seq%20&%20RNN%20927b8057e1e74a8ca8c608feb367c468/attention_process.mp4)