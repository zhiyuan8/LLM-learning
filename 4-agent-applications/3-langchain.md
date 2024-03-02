# Langchain

## Industry Application

The biggest developer community in GenAI

<img src="Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled.png" width="600"/>

builders choose LangChain

<img src="Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%201.png" width="300"/>

What does `langchain` mean?

*`Chains`* are the fundamental principle that holds various AI components in LangChain. A chain is a series of automated actions from the user's query to the model's output.

## Architecture

**LangChain** is a framework for developing applications powered by language models. It consists:

- **[LangSmith](https://python.langchain.com/docs/langsmith)**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.
- **[LangServe](https://python.langchain.com/docs/langserve)**: A library for deploying LangChain chains as a REST API.
- **[LangChain Templates](https://python.langchain.com/docs/templates)**: A collection of easily deployable reference architectures for a wide variety of tasks.
- **LangChain Libraries:**
    - **`langchain-core`**: Base abstractions and LCEL (LangChain Expression Language).
    - **`langchain-community`**: Third-party integrations.
    - **`langchain`**: **Chains**, **agents**, and **retrieval** strategies that make up an application's cognitive architecture.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%202.png)

# LangChain-Core

**LCEL** makes it easy to build complex chains from basic components.

The `|` symbol is similar to a [unix pipe operator](https://en.wikipedia.org/wiki/Pipeline_(Unix)), which chains together the different components feeds the output from one component as input into the next component.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%203.png)

More examples at [reference doc](https://python.langchain.com/docs/expression_language/why)

# LangChain-Community

## Module IO

Interface with language models

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%204.png)

1. **Prompt Templates**

few-shot prompt examples:

| Name | Description |
| --- | --- |
| Similarity | Uses semantic similarity between inputs and examples to decide which examples to choose. |
| MMR | Uses Max Marginal Relevance between inputs and examples to decide which examples to choose. |
| Length | Selects examples based on how many can fit within a certain length |
| Ngram | Uses ngram overlap between inputs and examples to decide which examples to choose. |
1. **Chat Model Integration**

[https://python.langchain.com/docs/integrations/chat/](https://python.langchain.com/docs/integrations/chat/)

| Model | Invoke | Async invoke | Stream | Async stream |
| --- | --- | --- | --- | --- |
| AzureChatOpenAI | ✅ | ✅ | ✅ | ✅ |
| ChatAnthropic | ✅ | ✅ | ✅ | ✅ |
| … | … | … | … | … |
1. **Output Parsers**

parse output in a specific format. using prompt engineering. The format can be csv, datetime, json, xml …

[https://python.langchain.com/docs/modules/model_io/output_parsers/types/csv](https://python.langchain.com/docs/modules/model_io/output_parsers/types/csv)

## Retrieval

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%205.png)

1. **Document Loader**

CSV, HTML, JSON, PDF .. loader

1. **Text Splitter**

Split the text up into small, semantically meaningful chunks

**Semantic Chunking:** This chunker works by determining when to “break” apart sentences. This is done by looking for differences in embeddings between any two sentences. When that difference is past some threshold, then they are split.

1. **Vector Store**

Chroma, LanceDB

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%206.png)

1. **Retrievers**

The `EnsembleRetriever` takes a list of retrievers as input and ensemble the results of their `get_relevant_documents()` methods and rerank the results

The most common pattern is to combine a sparse retriever (like BM25) with a dense retriever (like embedding similarity), because their strengths are complementary. It is also known as “hybrid search”. The sparse retriever is good at finding relevant documents based on keywords, while the dense retriever is good at finding relevant documents based on semantic similarity.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%207.png)

1. **Indexing**

## Agent Tooling

When constructing your own ***agent***, you will need to provide it with a list of ***Tools*** that it can use.

[https://python.langchain.com/docs/integrations/toolkits/](https://python.langchain.com/docs/integrations/toolkits/)

# LangChain

The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions - that is done by the AgentExecutor

## Agents

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%208.png)

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%209.png)

### OpenAI Agent

```jsx
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

Use CoT (Chain of Thought) : [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

*`chain of thought`*a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2010.png)

### XML Agent / Ahthropic Agent

```jsx
llm = ChatAnthropic(model="claude-2")
agent = create_xml_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### ReAct Agent

ReAct paper : [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

A ReAct prompt consists of few-shot task-solving trajectories, with human-written text reasoning traces and actions, as well as environment observations in response to actions

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2011.png)

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2012.png)

### Self-ask Agent / Reflexion

Reflexion paper : [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2013.png)

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2014.png)

## **Memory**

A chain will interact with its memory system twice in a given run.

1. AFTER receiving the initial user inputs but BEFORE executing the core logic, a chain will READ from its memory system and augment the user inputs.
2. AFTER executing the core logic but BEFORE returning the answer, a chain will WRITE the inputs and outputs of the current run to memory, so that they can be referred to in future runs.

![Untitled](Langchain%20bb0cede644314e0ab832ddaade270ffe/Untitled%2015.png)

# Review on Langchain

### from the technical side

- LangChain is an open-source tool for generative AI development.
- LangChain provides developers with a standardized and accessible way of building LLM-powered applications. It simplifies complex AI processes for “citizen data scientists.”
- LangChain offers abstractions and tools for Python and JavaScript packages, allowing developers to connect LLMs to data.
- Good support for agent & memory

### from the business side

- Very good community
    - [https://smith.langchain.com/hub](https://smith.langchain.com/hub)
    - [https://python.langchain.com/docs/people/](https://python.langchain.com/docs/people/)
- Demonstrate usages in different industries
    - [https://blog.langchain.dev/tag/case-studies/](https://blog.langchain.dev/tag/case-studies/)
    - [https://blog.langchain.dev/ai-powered-medical-knowledge/](https://blog.langchain.dev/ai-powered-medical-knowledge/)