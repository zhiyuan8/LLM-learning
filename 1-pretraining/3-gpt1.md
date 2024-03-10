
# GPT1

- GPT-1 (Generative Pre-trained Transformer) is a language model introduced by OpenAI in 2018. It is a transformer-based model that is pre-trained on a large corpus of text in an unsupervised manner using a language modeling objective. After the pre-training phase, the model can be fine-tuned on specific tasks using labeled data.
- GPT-1 is an `autoregressive model`, meaning it generates text one token at a time. After each token is produced, it is added to the input sequence, and the model predicts the next token based on the updated input.
- Unlike BERT, which uses regular self-attention, GPT-1 employs `masked self-attention`. This means that during the **self-attention** calculation, future tokens are masked (hidden) to maintain the autoregressive property.
- a `single task-agnostic model` through generative pre-training and discriminative fine-tuning.

## **Architecture**:

GPT-1 is a **decoder-only** transformer model, meaning it uses only the decoder part of the transformer architecture. It is **unidirectional**, processing the input text from left to right, unlike BERT, which is bidirectional, and BERT can predict.

**Unsupervised pre-training**

Given an unsupervised corpus of tokens $\mathcal{U} = \{u_1, \ldots, u_n\}$ we use a standard language modeling objective to maximize the following likelihood:

where \( k \) is the size of the context window, and the conditional probability \( P \) is modeled using a neural network with parameters \( \Theta \). These parameters are trained using stochastic gradient descent \cite{51}.

$$
L_1 (\mathcal{U}) = \sum_{i} \log P(u_i | u_{i-k}, \ldots, u_{i-1}; \Theta)
$$

we use a multi-layer *`Transformer decoder`* which is a variant of the transformer

$$
h_0 = U W_e + W_p
$$

$$
h_l = \text{transformer block}(h_{l-1}) \quad \forall l \in [1, n]
$$

$$
P(u) = \text{softmax}(h_n W^T_e)
$$

where $U = (u_{-k},\dots,u_{-1})$ is the context vector of tokens, n is the number of layers, $W_e$ is the token embedding matrix, and $W_p$ is the position embedding matrix.

**Supervised fine-tunning with two objective functions**
The inputs are passed through our pre-trained model to obtain the final transformer block’s activation  $h^m_l$, which is then fed into an added linear output layer with parameters $W_y$ to predict y:

$$
P(y|x^1, \ldots, x^m) = \text{softmax}(h^m_l W_y).
$$

This gives us the following objective to maximize:

$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y|x^1, \ldots, x^m).
$$

We additionally found that including language modeling as an auxiliary objective to the fine-tuning helped learning by (a) improving the generalization of the supervised model, and (b) accelerating convergence:

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \ast L_1(\mathcal{C}).
$$

**Task-specific input transformations**
All transformations include adding randomly initialized start and end tokens (⟨s⟩, ⟨e⟩).

![Untitled](GPT1%208b5848ee8ec54f4da0583831eb2e7a39/Untitled.png)

### Textual Entailment

- **Task Description**: Determine if the premise logically entails the hypothesis.
- **Example**:
    - **Premise**: "Dad gives flower to mom."
    - **Hypothesis**: "Dad loves mom."
    - **Explanation**: The model assesses whether the action described in the premise (giving a flower) implies the hypothesis (love). It does not necessarily prove love but could suggest it, so the task is to decide if one can logically infer the hypothesis from the premise.

### Similarity

- **Task Description**: Assess the degree of semantic similarity between two sentences.
- **Example**:
    - **Sentence 1**: "The cat sits on the mat."
    - **Sentence 2**: "A cat is sitting on a mat."
    - **Explanation**: Both sentences convey nearly identical meanings but with different wording. The model evaluates how similar the sentences are in meaning, potentially identifying them as paraphrases.

### Question Answering and Commonsense Reasoning

- **Task Description**: Select the correct answer from a set of options based on a given context and question.
- **Example**:
    - **Context (z)**: "Alice placed the cake in the fridge to keep it cool."
    - **Question (q)**: "Where is the cake?"
    - **Possible Answers $a_k$**:
        1. "In the oven."
        2. "On the table."
        3. "In the fridge."
    - **Explanation**: The model uses the context to determine that the correct answer is "In the fridge," demonstrating an understanding of the story's details and commonsense knowledge about where cakes can be stored to stay cool.

# References
- [Improving Language Understanding by Generative Pre-Training](https://scholar.google.com/scholar?q=Improving+Language+Understanding+by+Generative+Pre-Training+archiv&hl=en&as_sdt=0&as_vis=1&oi=scholart)
- [AndrejKarpathy GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)