# Prompt Engineering

- COT, Chain of Thought
- Few-shot Learning

## Best practices for chatGPT

From [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)

- Write Clear Instructions
    - [Ask the model to adopt a persona](https://platform.openai.com/docs/guides/prompt-engineering/tactic-ask-the-model-to-adopt-a-persona) : use system message
    - [Include details in your query to get more relevant answers](https://platform.openai.com/docs/guides/prompt-engineering/tactic-include-details-in-your-query-to-get-more-relevant-answers) : Who’s president? ⇒ Who was the president of Mexico in 2021
    - [Use delimiters to clearly indicate distinct parts of the input](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-delimiters-to-clearly-indicate-distinct-parts-of-the-input) , <article>
    - [Specify the steps required to complete a task](https://platform.openai.com/docs/guides/prompt-engineering/tactic-specify-the-steps-required-to-complete-a-task)
    - [Provide examples](https://platform.openai.com/docs/guides/prompt-engineering/tactic-provide-examples) : few-shot learning
    - [Specify the desired length of the output](https://platform.openai.com/docs/guides/prompt-engineering/tactic-specify-the-desired-length-of-the-output)
- **[Provide reference text](https://platform.openai.com/docs/guides/prompt-engineering/provide-reference-text)**
    - [Instruct the model to answer using a reference text](https://platform.openai.com/docs/guides/prompt-engineering/tactic-instruct-the-model-to-answer-using-a-reference-text)
    - [Instruct the model to answer with citations from a reference text](https://platform.openai.com/docs/guides/prompt-engineering/tactic-instruct-the-model-to-answer-with-citations-from-a-reference-text) : make reply indicate citations for human check
- **[Split complex tasks into simpler subtasks](https://platform.openai.com/docs/guides/prompt-engineering/split-complex-tasks-into-simpler-subtasks)**
    - [Use intent classification to identify the most relevant instructions for a user query](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-intent-classification-to-identify-the-most-relevant-instructions-for-a-user-query)
    - [For dialogue applications that require very long conversations, summarize or filter previous dialogue](https://platform.openai.com/docs/guides/prompt-engineering/tactic-for-dialogue-applications-that-require-very-long-conversations-summarize-or-filter-previous-dialogue)
    - [Summarize long documents piecewise and construct a full summary recursively](https://platform.openai.com/docs/guides/prompt-engineering/tactic-summarize-long-documents-piecewise-and-construct-a-full-summary-recursively)
- **[Give the model time to "think"](https://platform.openai.com/docs/guides/prompt-engineering/give-the-model-time-to-think)**
    - [Instruct the model to work out its own solution before rushing to a conclusion](https://platform.openai.com/docs/guides/prompt-engineering/tactic-instruct-the-model-to-work-out-its-own-solution-before-rushing-to-a-conclusion) : We can get the model to successfully notice this by prompting the model to generate its own solution first, before asking model whether the answer is correct or not.
    - [Use inner monologue or a sequence of queries to hide the model's reasoning process](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-inner-monologue-or-a-sequence-of-queries-to-hide-the-model-s-reasoning-process) : Inner monologue is a tactic that can be used to mitigate this. The idea of inner monologue is to instruct the model to put parts of the output that are meant to be hidden from the user into a structured format that makes parsing them easy.
    - [Ask the model if it missed anything on previous passes](https://platform.openai.com/docs/guides/prompt-engineering/tactic-ask-the-model-if-it-missed-anything-on-previous-passes)
- **[Use external tools](https://platform.openai.com/docs/guides/prompt-engineering/use-external-tools)**
    - [Use embeddings-based search to implement efficient knowledge retrieval](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-embeddings-based-search-to-implement-efficient-knowledge-retrieval)
    - [Use code execution to perform more accurate calculations or call external APIs](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-code-execution-to-perform-more-accurate-calculations-or-call-external-apis)
    - [Give the model access to specific functions](https://platform.openai.com/docs/guides/prompt-engineering/tactic-give-the-model-access-to-specific-functions)

[Use delimiters to clearly indicate distinct parts of the input](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-delimiters-to-clearly-indicate-distinct-parts-of-the-input)

```markdown
SYSTEM
You will be provided with a pair of articles (delimited with XML tags) about the same topic. First summarize the arguments of each article. Then indicate which of them makes a better argument and explain why.

USER
<article> insert first article here </article>
<article> insert second article here </article>
```

[Specify the steps required to complete a task](https://platform.openai.com/docs/guides/prompt-engineering/tactic-specify-the-steps-required-to-complete-a-task)

```markdown
SYSTEM
Use the following step-by-step instructions to respond to user inputs.
Step 1 - The user will provide you with text in triple quotes. Summarize this text in one sentence with a prefix that says "Summary: ".
Step 2 - Translate the summary from Step 1 into Spanish, with a prefix that says "Translation: ".

USER
"""insert text here"""
```

[Instruct the model to answer with citations from a reference text](https://platform.openai.com/docs/guides/prompt-engineering/tactic-instruct-the-model-to-answer-with-citations-from-a-reference-text)

```markdown
SYSTEM
You will be provided with a document delimited by triple quotes and a question. Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question. If the document does not contain the information needed to answer this question then simply write: "Insufficient information." If an answer to the question is provided, it must be annotated with a citation. Use the following format for to cite relevant passages ({"citation": …}).
USER
"""<insert document here>"""

Question: <insert question here>
```

[Use inner monologue or a sequence of queries to hide the model's reasoning process](https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-inner-monologue-or-a-sequence-of-queries-to-hide-the-model-s-reasoning-process): use COT

```markdown
SYSTEM

Follow these steps to answer the user queries.
Step 1 - First work out your own solution to the problem. Don't rely on the student's solution since it may be incorrect. Enclose all your work for this step within triple quotes (""").
Step 2 - Compare your solution to the student's solution and evaluate if the student's solution is correct or not. Enclose all your work for this step within triple quotes (""").
Step 3 - If the student made a mistake, determine what hint you could give the student without giving away the answer. Enclose all your work for this step within triple quotes (""").
Step 4 - If the student made a mistake, provide the hint from the previous step to the student (outside of triple quotes). Instead of writing "Step 4 - ..." write "Hint:".

USER
Problem Statement: <insert problem statement>

Student Solution: <insert student solution>
```

## Best practices for OpenAI API

- Put instructions at the beginning of the prompt and **use ### or """** to separate the instruction and context
- Articulate the desired output format through examples
- Instead of just saying what not to do, say what to do instead
- **Code Generation Specific -** Use “leading words” to nudge the model toward a particular pattern
- For most **factual use cases** such as data extraction, and truthful Q&A, the `temperature` of **0** is best.

Use ### or """ to separate the instruction and context 

```markdown
Summarize the text below as a bullet point list of the most important points.

Text: """
{text input here}
"""
```

# References

- [Meet Notes](https://docs.google.com/document/d/1B0Xhyyqgz_fqjj7e7jD_eIK1ByygM48VJyL5J_zEByI/edit#heading=h.yv7qylxozth0)
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt engineering with the OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)