# Untitled

Mainly from **[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives?utm_source=profile&utm_medium=reader2)**

# Reinforcement learning

Reinforcement learning is a [machine learning](https://www.techtarget.com/searchenterpriseai/definition/machine-learning-ML) training method based on rewarding desired behaviors and punishing undesired ones. In general, a reinforcement learning agent -- the entity being trained -- is able to perceive and interpret its environment, take actions and learn through trial and error.

# Supervised fine tuning cons:

models are incentivized to place probability mass on all human demonstrations, including those that are low-quality; and distributional shift during sampling can degrade performance

# RLHF goal

The essential goal here is to make a conventional large language model (GPT-3 in our case) align with human principles or preferences. This makes our LLMs less toxic, more truthful, and less biased.

# Steps:

[https://lh7-us.googleusercontent.com/-lRmhVw8q5oCVVu57Un7o6sWF6MOiG2DV-_kd27pO4wSJBQW-7D0o6uANaGv1ef7ljQ0SXbL1e4lNRPZChsKl1iFbuyHHiqazJt8ynjCPDxoRbh3PMwnOeXvWewcj_XeZUYsMKswKVOOMtRCCq0pBg](https://lh7-us.googleusercontent.com/-lRmhVw8q5oCVVu57Un7o6sWF6MOiG2DV-_kd27pO4wSJBQW-7D0o6uANaGv1ef7ljQ0SXbL1e4lNRPZChsKl1iFbuyHHiqazJt8ynjCPDxoRbh3PMwnOeXvWewcj_XeZUYsMKswKVOOMtRCCq0pBg)

1. Pretraining a language model (LM),
    1. fine tune with preference data set or supervised learning
    2. supervised fine tuning uses a smaller dataset
2. gathering data and training a reward model, and
    1. These LMs for reward modeling can be both another fine-tuned LM or a LM trained from scratch on the preference data.
        1. fine-tuned LM’s output layer (the next-token classification layer) is substituted with a regression layer, which features a single output node.
    2. sample a set of prompts from a predefined dataset
    3. pass through the initial language model to generate new text.
    4. for each prompt, we generate four to nine responses from the finetuned LLM created in the prior step. An individual then ranks these responses based on their preference.
    5. Human annotators are used to rank the generated text outputs from the LM
        1. There are multiple methods for ranking the text. One method that has been successful is to have users compare generated text from two language models conditioned on the same prompt. By comparing model outputs in head-to-head matchups, an **[Elo](https://en.wikipedia.org/wiki/Elo_rating_system)** system can be used to generate a ranking of the models and outputs relative to each-other. These different methods of ranking are normalized into a scalar reward signal for training.
    6. An interesting artifact of this process is that the successful RLHF systems to date have used reward language models with varying sizes relative to the text generation (e.g. OpenAI 175B LM, 6B reward model, Anthropic used LM and reward models from 10B to 52B, DeepMind uses 70B Chinchilla models for both LM and reward).
    7. An intuition would be that these preference models need to have similar capacity to understand the text given to them as a model would need in order to generate said text.
3. fine-tuning the LM with reinforcement learning.
    1. fine-tuning some or all of the parameters of a **copy of the initial LM** with a policy-gradient RL algorithm, Proximal Policy Optimization (PPO)
    2. Some parameters of the LM are frozen

# InstructGPT stats

- Pretrain: 100B - 5T tokens
- Supervised Finetuning: 1k - 50k instruction response pairs
- RLHF > 50k examples

# RLHF in Llama 2

[https://lh7-us.googleusercontent.com/wL4rYvpZp7S8N4a_dOCZOfwmoJgd47nBtZCfbI2Wb34Y7FrcYLT8QnJ3T8FgG4OlwL_JKff1IDD84VLTvy4GTkoR-Oq6YvFFsRJKhY-jaVF3ymFbyot-wjPjXwEPP5jz8Ej0uzmbqFAFCfv_SguX-Q](https://lh7-us.googleusercontent.com/wL4rYvpZp7S8N4a_dOCZOfwmoJgd47nBtZCfbI2Wb34Y7FrcYLT8QnJ3T8FgG4OlwL_JKff1IDD84VLTvy4GTkoR-Oq6YvFFsRJKhY-jaVF3ymFbyot-wjPjXwEPP5jz8Ej0uzmbqFAFCfv_SguX-Q)

- 2 reward model
    - Helpfulness
    - Harmlessness
- Different human ranking method
    - InstructGPT ask human labelers to rank 4 responses at a time
    - LLama 2 only presents 2 responses for ranking but an additional "margin" label (ranging from "significantly better" to "negligibly better") is gathered
- Different ranking loss function to train reward model
    - InstructGPT loss
        
        [https://lh7-us.googleusercontent.com/BeK1RO2F0T8i2m0kbn1xdNs4zoHm_HFOCgPI_GdnA9UyNTJH4zECC0mzwb-yLc7KOf1EuL3Lm3WlR5oRg5uE_YkN1B2yR8vaAfA_7m9rUpbuZJ1xNSBxoRhnghlw74Jlllaq3y4yEo8gc2p5n_QOZg](https://lh7-us.googleusercontent.com/BeK1RO2F0T8i2m0kbn1xdNs4zoHm_HFOCgPI_GdnA9UyNTJH4zECC0mzwb-yLc7KOf1EuL3Lm3WlR5oRg5uE_YkN1B2yR8vaAfA_7m9rUpbuZJ1xNSBxoRhnghlw74Jlllaq3y4yEo8gc2p5n_QOZg)
        
    - Llama2 loss
        
        [https://lh7-us.googleusercontent.com/-qDkjLPzHIXgRNUyJ9gZKaEttWupbtQAuSp64z8t7MZhKW5WCl_eGTIZSyrl7rEgcuqHSBple8HRBsLlnGoY-tuQm6Js8aCt8NyfxjrEUYtxvvPRUKFH1EOvvVhHs1LhWswz7rWQmQyw6joM7AAvHQ](https://lh7-us.googleusercontent.com/-qDkjLPzHIXgRNUyJ9gZKaEttWupbtQAuSp64z8t7MZhKW5WCl_eGTIZSyrl7rEgcuqHSBple8HRBsLlnGoY-tuQm6Js8aCt8NyfxjrEUYtxvvPRUKFH1EOvvVhHs1LhWswz7rWQmQyw6joM7AAvHQ)
        
    - Llama 2 added the the margin “m(r)” as a discrete function of the preference rating as follows:
    - returning a higher margin via “m(r)” will make the difference between the reward of the preferred and rejected responses smaller, resulting in a larger loss, which in turn results in larger gradients, and consequently model changes, during the policy gradient update.
- 2 RLHF stages
    - Rejection sampling
        - K outputs are drawn, and the one with the highest reward is chosen for the gradient update during the optimization step
    - PPO stage

# Proximal Policy Optimization Algorithms

[https://lh7-us.googleusercontent.com/wLt3I7wtaVVYBd_SVMYZTW47mpjUO18kx_M_UZVFm1v69_QoOOqpsKDcZi7tVfyFV2dHG4M15CeOlUPTZ4_GhpFzEqozoWT3V9jn3Hqe4vGVGUtgrKYyNIoo4CgPXn5YQO3pk8YgGHcpBB7BSL8ofw](https://lh7-us.googleusercontent.com/wLt3I7wtaVVYBd_SVMYZTW47mpjUO18kx_M_UZVFm1v69_QoOOqpsKDcZi7tVfyFV2dHG4M15CeOlUPTZ4_GhpFzEqozoWT3V9jn3Hqe4vGVGUtgrKYyNIoo4CgPXn5YQO3pk8YgGHcpBB7BSL8ofw)

[https://lh7-us.googleusercontent.com/bP4K6oiTFYpDfb_U0uwidlptiGXoZxQBR9GGSPEBEBue8VzPNXV1Fi2lCxbRqv9QxsiPmdyBBUyKq7C6Nks0JNcAm-K5DY4qYIOnMq3XIH56H_oga4KZuxuyqac51RKGpfGQhIiKT45Upn52Hwly9g](https://lh7-us.googleusercontent.com/bP4K6oiTFYpDfb_U0uwidlptiGXoZxQBR9GGSPEBEBue8VzPNXV1Fi2lCxbRqv9QxsiPmdyBBUyKq7C6Nks0JNcAm-K5DY4qYIOnMq3XIH56H_oga4KZuxuyqac51RKGpfGQhIiKT45Upn52Hwly9g)

1. Given a prompt, *x*, from the dataset, the text *y* is generated by the current iteration of the fine-tuned policy.
2. Concatenated with the original prompt, that text is passed to the preference model, which returns a scalar notion of “preferability”, *rθ*.
3. In addition, per-token probability distributions from the RL policy are compared to the ones from the initial model to compute a penalty on the difference between them.
    1. In multiple papers from OpenAI, Anthropic, and DeepMind, this penalty has been designed as a scaled version of the Kullback–Leibler **[(KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)** between these sequences of distributions over tokens, Rkl.
    2. The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, which can be useful to make sure the model outputs reasonably coherent text snippets.
    3. Without this penalty the optimization can start to generate text that is gibberish but fools the reward model to give a high reward.
    4. In practice, the KL divergence is approximated via sampling from both distributions (explained by John Schulman **[here](http://joschu.net/blog/kl-approx.html)**). The final reward sent to the RL update rule is *r*=*rθ*−*λ*Rkl.
        1. This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode.
        2. Second, it ensures the policy doesn’t learn to produce outputs that are too different from those that the reward model has seen during training.
4. the **update rule** is the parameter update from PPO that maximizes the reward metrics in the current batch of data (PPO is on-policy, which means the parameters are only updated with the current batch of prompt-generation pairs). PPO is a trust region optimization algorithm that uses constraints on the gradient to ensure the update step does not destabilize the learning process.

## Results for RLHF with PPO:

- better than supervised learning in summarizing reddit posts and news articles
- over optimize the reward model hurt the true preference on llm output
- doubling the training data amount leads to a ~1.1% increase in the reward model validation set accuracy, whereas doubling the model size leads to a ~1.8% increase
- our reward models are sensitive to small but semantically important details in the summary.
- our learned reward models consistently outperform other metrics such as ROUGE, summary length, amount of copying from the post, and log probability under our baseline supervised models.

[https://lh7-us.googleusercontent.com/Nep6nWumq77-_wa4H-7ey3SVQG267QBFAj7gpU8b0OmeW3EatPmbCNpBpGAK9-4Qi0i-7spNi-glEqtWbHMHGj-CpK4xcVTCQoLS9SE2CPuiAqpyJvPtIWRcLlKwB2P_h-VWgJth0G0oNMR5RkmmDw](https://lh7-us.googleusercontent.com/Nep6nWumq77-_wa4H-7ey3SVQG267QBFAj7gpU8b0OmeW3EatPmbCNpBpGAK9-4Qi0i-7spNi-glEqtWbHMHGj-CpK4xcVTCQoLS9SE2CPuiAqpyJvPtIWRcLlKwB2P_h-VWgJth0G0oNMR5RkmmDw)

# Direct Preference Optimization:

**RLHF cons**

- More complex
- High computational cost

In this paper, we show how to directly optimize a language model to adhere to human preferences, without explicit reward modeling or reinforcement learning.

**Given a dataset of human preferences over model responses, DPO can therefore optimize a policy using a simple binary cross entropy objective, producing the optimal policy to an implicit reward function fit to the preference data.**

our key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies. This change-of-variables approach avoids fitting an explicit, standalone reward model, while still optimizing under existing models of human preferences, such as the Bradley-Terry model. In essence, the policy network represents both the language model and the (implicit) reward.

[https://lh7-us.googleusercontent.com/fsCAYjdH1bs_IvPt0RR5ZQ0ZfyZ-ZVxE_CajyJPmjUrAN4KKiLaCUb85t_2e00a-QFsOiPzDiwFARJOndEL3ZUII2r9RAD-nh4UpZUc_l05e6egRHW-BVOtwl0NjOVMARxV2yLbHZhCR5ZCvfzUTRA](https://lh7-us.googleusercontent.com/fsCAYjdH1bs_IvPt0RR5ZQ0ZfyZ-ZVxE_CajyJPmjUrAN4KKiLaCUb85t_2e00a-QFsOiPzDiwFARJOndEL3ZUII2r9RAD-nh4UpZUc_l05e6egRHW-BVOtwl0NjOVMARxV2yLbHZhCR5ZCvfzUTRA)

[https://lh7-us.googleusercontent.com/BvvKnCkHTA4Ondt9xnOcxMRbwLIcd4Yhj419hScTkDmooYFV21l8K_bPxLKKHqU9b8rJjN6u2vDV8Bc2hqx3EBloa2hdP5I7U5qUxEmr7ffgHKWdz4eV4zdGao1mVeGlDsQIu-ntBBRohxjoh3KThw](https://lh7-us.googleusercontent.com/BvvKnCkHTA4Ondt9xnOcxMRbwLIcd4Yhj419hScTkDmooYFV21l8K_bPxLKKHqU9b8rJjN6u2vDV8Bc2hqx3EBloa2hdP5I7U5qUxEmr7ffgHKWdz4eV4zdGao1mVeGlDsQIu-ntBBRohxjoh3KThw)

[https://lh7-us.googleusercontent.com/GHBGVQGHAtZkKA5UxdP3HzIGxj3F2HCW_PhfMs9D2V5GeZIS274wK54K3wHsQrzL5dcoLI8HkwlfayqrhlRBkdEG6CrrWPI0SSzJDH-R0MCpHpw2smi6HVETmFC9YT0lBoY2RcBO9Qd_s5_IYElPtg](https://lh7-us.googleusercontent.com/GHBGVQGHAtZkKA5UxdP3HzIGxj3F2HCW_PhfMs9D2V5GeZIS274wK54K3wHsQrzL5dcoLI8HkwlfayqrhlRBkdEG6CrrWPI0SSzJDH-R0MCpHpw2smi6HVETmFC9YT0lBoY2RcBO9Qd_s5_IYElPtg)

[https://lh7-us.googleusercontent.com/iFotqIA38oiCG7QwwAMWO7-5jKV7-eqoe2Wpy_CO7VfHsBY0KwCwsmJDJbHKSuZS_ajvdO8Mkn9PJTwz33SxmqnBdhv0_CvyYxwzdED0-1FdMDXArwMLJmxZe4QuLCZQDVbEv1ozQwOSBPsQBpmQyw](https://lh7-us.googleusercontent.com/iFotqIA38oiCG7QwwAMWO7-5jKV7-eqoe2Wpy_CO7VfHsBY0KwCwsmJDJbHKSuZS_ajvdO8Mkn9PJTwz33SxmqnBdhv0_CvyYxwzdED0-1FdMDXArwMLJmxZe4QuLCZQDVbEv1ozQwOSBPsQBpmQyw)

**Experiments and Results**

- DPO converges to its best performance relatively quickly.
- DPO policies can generalize similarly well to PPO policies, even though DPO does not use the additional unlabeled Reddit TL;DR prompts that PPO uses.
- These experiments are judged by GPT-4 e.g. GPT-4 decides if a completion is better than human written summaries
- And another experiment, We find that with both prompts, GPT-4 tends to agree with humans about as often as humans agree with each other, suggesting that GPT-4 is a reasonable proxy for human evaluations

# RLAIF

## Constitutional AI: Harmlessness from AI Feedback (Dec 2022, [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073))

[https://lh7-us.googleusercontent.com/RioHUAe3f8Ewar1JPEfYoE5tQf1RP9mxbeJ7N2YW13yNqV71Js_2wu3Dd9TMnBzGQwIN4rD9vEj6blZLKzHYqfbks6Ot2GIsiALJeFKBcSGzQtZe88NPWvL3FCMxVfWqHiEetYe4Nuu4rpaPJrJc4Q](https://lh7-us.googleusercontent.com/RioHUAe3f8Ewar1JPEfYoE5tQf1RP9mxbeJ7N2YW13yNqV71Js_2wu3Dd9TMnBzGQwIN4rD9vEj6blZLKzHYqfbks6Ot2GIsiALJeFKBcSGzQtZe88NPWvL3FCMxVfWqHiEetYe4Nuu4rpaPJrJc4Q)

[https://lh7-us.googleusercontent.com/1ud73jSyLdPvgiT4PD7GEm0B_z2WvS5clZgj7gKmC1gw_pR0ILQgHXQjnqAE4zfNlYHiSldmfla6kcFniCStNG800b8juoLFrZ_bbxdA2cgwp7QZJbH3hgzD2JYL5gm5V7kSP896d_X4KjrlGB9B6Q](https://lh7-us.googleusercontent.com/1ud73jSyLdPvgiT4PD7GEm0B_z2WvS5clZgj7gKmC1gw_pR0ILQgHXQjnqAE4zfNlYHiSldmfla6kcFniCStNG800b8juoLFrZ_bbxdA2cgwp7QZJbH3hgzD2JYL5gm5V7kSP896d_X4KjrlGB9B6Q)

## **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** (Sep 2023, [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267))

The main contributions of this work are as follows:

1. We demonstrate that RLAIF achieves comparable or superior performance to RLHF on the tasks of summarization, helpful dialogue generation, and harmless dialogue generation.
2. We show that RLAIF can improve upon a SFT policy even when the LLM labeler is the same size as the policy.
3. We find that directly prompting the LLM for reward scores during RL can outperform the canonical setup where a reward model is trained on LLM preferences.
4. We compare various techniques for generating AI labels and identify optimal settings for RLAIF practitioners.
    1. use chain of thoughts prompting and few shot prompting

[https://lh7-us.googleusercontent.com/Aj21xQdup9hIt2qqUi4GqhpJyc7YgK7t7350OjTNGXEh94yYaSDYcuJRCeGFpaYzdInSniIuUi5FSWbZJbHZuimMTNZk-luckU9hzhfYoz_os8LryCnQ7QTNxa42t8UxP3RcxayRAJKt5Hc67aUqGg](https://lh7-us.googleusercontent.com/Aj21xQdup9hIt2qqUi4GqhpJyc7YgK7t7350OjTNGXEh94yYaSDYcuJRCeGFpaYzdInSniIuUi5FSWbZJbHZuimMTNZk-luckU9hzhfYoz_os8LryCnQ7QTNxa42t8UxP3RcxayRAJKt5Hc67aUqGg)

[https://lh7-us.googleusercontent.com/fwrs7uXgxCpdJPUex6YwsGthkN91TO6bpKHwj4-0qEdDmH0DVUqffJ71pXawpVmkD4pmtqV2X-5RXJkhzXijnVJR7sA_qNlzqjpFXJ53aHWGCRsOdnDgqhP8uuBbijfEvH_u9QiR3ONrHr1tEUzsGg](https://lh7-us.googleusercontent.com/fwrs7uXgxCpdJPUex6YwsGthkN91TO6bpKHwj4-0qEdDmH0DVUqffJ71pXawpVmkD4pmtqV2X-5RXJkhzXijnVJR7sA_qNlzqjpFXJ53aHWGCRsOdnDgqhP8uuBbijfEvH_u9QiR3ONrHr1tEUzsGg)

2 approaches:

1. Distilled RLAIF: produces soft labels (e.g. [0.6, 0.4]), and train a reward model on it
2. Direct RLAIF: ask LLM model to rate from 1 - 10

Evaluation:

- AI Labeler Alignment measures the accuracy of AI-labeled preferences with respect to human preferences.
- Win Rate evaluates the end-to-end quality of two policies by measuring how often one policy is preferred by human annotators over another.
- Harmless Rate measures the percentage of responses that are considered harmless by human evaluators

Results

- RLAIF achieves performance gains on par with or better than RLHF on all three tasks
- One natural question that arises is whether there is value in combining human and AI feedback. We experimented with combining both types of feedback but did not see an improvement beyond using human feedback alone.
- RLAIF can yield improvements even when the AI labeler model is the same size (in terms number of params ) as the policy LLM.
    - We note that the AI labeler and initial policy are not the exact same model.
- Direct RLAIF performs better than Distilled RLAIF
    - One hypothesis for the improved quality is that bypassing the distillation from AI preferences into a RM enables information to flow directly from the off-the-shelf LLM to the policy.

[https://lh7-us.googleusercontent.com/gSV74srwvsMybmAi7JiNH6pX5es6b8Al11oxX6xe9eT-aA3b6S2fdbq-da12dF6ML8nvy23wsJcI8sgd_v2qwxoNmBW0DGQEW9cG7Zjy6nRlRzoiPcd53xCE3IGzs-TatVD-TNbxI65xUDEnDF7VdQ](https://lh7-us.googleusercontent.com/gSV74srwvsMybmAi7JiNH6pX5es6b8Al11oxX6xe9eT-aA3b6S2fdbq-da12dF6ML8nvy23wsJcI8sgd_v2qwxoNmBW0DGQEW9cG7Zjy6nRlRzoiPcd53xCE3IGzs-TatVD-TNbxI65xUDEnDF7VdQ)

- We observe that eliciting chain-of-thought reasoning generally improves AI labeler alignment, while the impacts of preamble specificity and in-context learning vary across tasks
- We also conduct experiments with selfconsistency (Wang et al., 2022b), where multiple chain-of-thought rationales are sampled with temperature T > 0. The preference distributions generated by the LLM are averaged together to arrive at the final preference label. We find that **selfconsistency** strictly degrades AI labeler alignment

[https://lh7-us.googleusercontent.com/PiogIOY9wTcgNWKAo0DrhIp_DrvvxczEMdZxhhoSMYy8VAqMi3BuYn4ixmxbpxLObbistzT8X0q6Ha2uHoIJJgUwSZ0sdnQ23x4ZJP_9iGkDAgJQPFQPYrtIXngZpXFxvcze6rlHLmDQunLnAOiavA](https://lh7-us.googleusercontent.com/PiogIOY9wTcgNWKAo0DrhIp_DrvvxczEMdZxhhoSMYy8VAqMi3BuYn4ixmxbpxLObbistzT8X0q6Ha2uHoIJJgUwSZ0sdnQ23x4ZJP_9iGkDAgJQPFQPYrtIXngZpXFxvcze6rlHLmDQunLnAOiavA)

- Results show that the policy trained with more aligned AI labels achieves a significantly higher win rate.
- larger ai labeler model size leads to better ai labeler alignment and produce even higher quality preference labels.
    - Since the AI labeler is only used to generate preference examples once and is not called during RL, using an even larger AI labeler is not necessarily prohibitively expensive.

# Other Optimization options

## The Wisdom of Hindsight Makes Language Models Better Instruction Followers

[https://lh7-us.googleusercontent.com/kF6zvigtBN_r_GBj4E0gJk0Qo2QHXq-t0HGF47a-Gt0n5Sk3EIFJWgsAf6PNsGyi5v71Bx4ezOxacHVG9kCUjwBWzAqTbQgh4UrvRavayPc4Lt5IoSTTmCo6V7XQq8m7VHaZ9-PmsIa5it70pOTAwg](https://lh7-us.googleusercontent.com/kF6zvigtBN_r_GBj4E0gJk0Qo2QHXq-t0HGF47a-Gt0n5Sk3EIFJWgsAf6PNsGyi5v71Bx4ezOxacHVG9kCUjwBWzAqTbQgh4UrvRavayPc4Lt5IoSTTmCo6V7XQq8m7VHaZ9-PmsIa5it70pOTAwg)

**Contrastive Preference Learning: Learning from Human Feedback without RL** (Oct 2023, [https://arxiv.org/abs/2310.13639](https://arxiv.org/abs/2310.13639))

Similar to DPO but used in robotics environment

**(5) Reinforced Self-Training (ReST) for Language Modeling** (Aug 2023, [https://arxiv.org/abs/2308.08998](https://arxiv.org/abs/2308.08998))

[https://lh7-us.googleusercontent.com/iwYd8MFaI1mpKRKpVhJevdQtR2MjXwrc5WVcSf_MmwyK-GQggfbfWgmGJZZgolqqL1NlFA6vapNPnyuGxrMwvOAVdnrdoHozwvSLE5u6svbZj5F7thosCE7QmAq-PEhJpU0gBoH7N6z5N6TaX9NTBw](https://lh7-us.googleusercontent.com/iwYd8MFaI1mpKRKpVhJevdQtR2MjXwrc5WVcSf_MmwyK-GQggfbfWgmGJZZgolqqL1NlFA6vapNPnyuGxrMwvOAVdnrdoHozwvSLE5u6svbZj5F7thosCE7QmAq-PEhJpU0gBoH7N6z5N6TaX9NTBw)

# References

- **[An Introduction to Training LLMs Using Reinforcement Learning From Human Feedback (RLHF)](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy)**
- **[Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)**
- **[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives?utm_source=profile&utm_medium=reader2)**
- **Learning to summarize from human feedback** [https://arxiv.org/pdf/2009.01325.pdf](https://arxiv.org/pdf/2009.01325.pdf)
- Proximal Policy Optimization Algorithms [https://arxiv.org/pdf/1707.06347.pdf](https://arxiv.org/pdf/1707.06347.pdf)
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267.pdf)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)