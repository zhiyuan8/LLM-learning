# **Toolformers**

[<**Toolformer: Language Models Can Teach Themselves to Use Tools**>](https://arxiv.org/abs/2302.04761)

*[Submitted on 9 Feb 2023]*

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled.png)

For each sentence: 

1. find a fitting place to call an API

It selects valid candidates by computing, for each slot *i*, the probability that the model M selects to start an API call at position *i*, i.e. the probability of predicting the <API> token. If this probability is below a threshold τ, the candidate is discarded.

```bash
[1] Pittsburgh [2] is [3] also [4] known [5] as [6] the [7] Steel [8] City [9]
```

1.  sample multiple potential inputs for the API 

We have up to m *ready* API calls for each position *i* 

```bash
[P(x), Pittsburgh, is, also, known, as, <API>]
```

 2. execute the API for each input

```bash
[P(x), Pittsburgh, is, also, known, as, <API>, QA(“What other name is Pittsburgh know by?”), →, Steel, City, </API>]
```

1. compare
- loss **including the API call and its results** as a prefix
- the minimum of the losses for **(i) doing no API call at all** and **(ii) doing an API call, but not providing the response**.

$$
L_i^+ = L_i(e(c_i, r_i))

$$

$$
L_i^- = \min(L_i(\epsilon), L_i(e(c_i, \epsilon)))
$$

we want the addition of the API call with its response to reduce the loss by at least a filtering threshold τ*_f*

$$
L_i^- - L_i^+ \geq \tau_f
$$

# **A**utomatic **R**easoning and **T**ool-use (ART)

[<**ART: Automatic multi-step reasoning and tool-use for large language models**>](https://arxiv.org/abs/2303.09014)

*[Submitted on 16 Mar 2023]*

LLMs can generate CoT-style multi-step reasoning in a zero-shot manner, when prompted with the prefix `“Let’s think step-by-step"` 

# **Gorilla**

[<**Gorilla: Large Language Model Connected with Massive APIs>**](https://arxiv.org/abs/2305.15334)

[Submitted on 24 May 2023]

Gorilla, finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls. Gorilla supports two modes - with retrieval, and `zero-shot`. 

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%201.png)

**Retriever-Aware training**

For training with retriever, the instruction-tuned dataset, also has an additional "Use this API documentation for reference: <retrieved_API_doc_JSON>" appended to the user prompt.

**Gorilla Inference**

- In zero-shot, this prompt (with NO further prompt tuning) is fed to the Gorilla LLM model when then returns the API call that will help in accomplishing the task and/or goal.
- In retrieval mode, the retriever (either of BM25 or GPT-Index) first retrieves the most up-to-date API documentation stored in the API Database.

**AST sub-tree matching**

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%202.png)

# **ToolAlpaca**

[<**ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases**>](https://arxiv.org/abs/2306.05301)

[Submitted on 8 Jun 2023 ([v1](https://arxiv.org/abs/2306.05301v1)), last revised 7 Sep 2023 (this version, v2)]

[github repo](https://github.com/tangqiaoyu/ToolAlpaca) 

- ToolAlpaca : a simple framework forthe automated generation of tool-use `corpus`
- each tool-use instance can be represent as a triple {*Instruction, Actions, Response*}

# Nexus-Raven

[Nexaus-Raven Github](https://github.com/nexusflowai/NexusRaven)

[Nexaus-Raven Paper](https://openreview.net/forum?id=Md6RUrGz67)

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%203.png)

- Chain-of-Thought Enhancement
- Hard-Negative Candidate Function List Generation

# ToolLLM / ToolLLaMA

[<ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs>](https://arxiv.org/abs/2307.16789)

[Submitted on 31 Jul 2023 ([v1](https://arxiv.org/abs/2307.16789v1)), last revised 3 Oct 2023 (this version, v2)]

- `ToolBench`:  an instruction-tuning dataset for tool use, which is constructed automatically using ChatGPT.
- To enhance the reasoning capabilities of LLMs, we develop a novel `depth-first search-based decision tree algorithm`. It enables LLMs to evaluate multiple reasoning traces and expand the search space.
- `ToolLLaMA`: with a neural API retriever, strong zero-shot generalization ability
    - Multi-tool
    - Multi-step reasoning

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%204.png)

## **Depth-first search-based decision tree** (DFSDT)

A general decision-making strategy to enhance the reasoning capabilities of LLMs.

DFSDT broadens the search space by considering multiple reasoning traces and achieves significantly better performance than ReACT.

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%205.png)


Task Description of Multi-tool Instructions:
```
You will be provided with several tools, tool descriptions, all of each tool’s available API functions, the descriptions of these API functions, and the parameters required for each API function. Your task involves creating 10 varied, innovative, and detailed user queries that employ API functions of multiple tools. 

For instance:
...

Deliver your response in this format: [Query1: ......, ‘related apis’:[[tool name, api name], [tool name, api name], [tool name, api name]...],Query2: ......, ‘related apis’:[[tool name, api name], [tool name, api name], [tool name, api name]...] ...]
```
In-context Seed Examples:
```
“Query”: “For my best friend’s surprise birthday party, I require inspiration for party games and decorations. Kindly suggest some random words that can serve as themes for the party. Furthermore, I’m interested in gathering news articles about the latest party trends to ensure a modern celebration. Also, I would appreciate details about the local hotels in my area for accommodation options. Your assistance is greatly appreciated.”, “related apis”: [[’Random Words’, ‘Get multiple random words’], [’thedigitalnewsfeederapi’, ‘Getting News Articles’], [’thedigitalnewsfeederapi’, ‘Getting all news articles’]]
```
Solution Path Annotation: Use the following prompt when searching for the solution path. When expanding the child nodes, we use diversity user prompt, showing the information of previous child nodes.
```
You are Tool-GPT, capable of utilizing numerous tools and functions to complete the given task.

1. First, I will provide you with the task description, and your task will commence.

2. At each step, you need to analyze the current status and determine the next course of action by executing a function call.

3. Following the call, you will receive the result, transitioning you to a new state. Subsequently, you will analyze your current status, make decisions about the next steps, and repeat this process.

4. After several iterations of thought and function calls, you will ultimately complete the task and provide your final answer.

Task description: {task_description}

---------------------------------------------------------
diversity_user_prompt:
{previous_candidate}

---------------------------------------------------------
Finish_function_description:
def Finish()
	'''
	return_type(enum): ["give_answer","give_up_and_restart"]
	'''
```

## ToolLLaMA - Github

You can use the following command to train ToolLLaMA-7b with **2 x A100 (80GB)**, with our preprocessed data:

[training tutorial](https://github.com/OpenBMB/ToolBench)

# **APISERVE**

[<**APIServe: Efficient API Support for Large-Language Model Inferencing**>](https://arxiv.org/abs/2402.01869)

APISERVE improves the overall serving throughput by 1.6× and completes 2× more requests per second.

![Untitled](LLM%20Tool%20Usages%2099c1ec5adb174d11936634a4c3ee3721/Untitled%206.png)

# Reference
- [Toolformers medium](https://medium.com/@boris.meinardus/toolformer-giving-large-language-models-tools-1562c3bf69fb)
- [Toolformers medium 2](https://www.shaped.ai/blog/breaking-down-toolformer)
- [Toolformer Github](https://github.com/conceptofmind/toolformer)
- [Toolformer Pytorch](https://github.com/lucidrains/toolformer-pytorch/tree/main)