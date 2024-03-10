# GPT4
- **Predictable Scaling**
    - reliably predict some aspects of the performance of GPT-4 from smaller models trained using 1, 000× – 10, 000× less compute.
    - `Power of Law`
        - The metric is final loss on a dataset derived from our internal codebase. We chose to look at loss because it tends to be less noisy than other measures across different amounts of training compute. A power law fit to the smaller models (excluding GPT-4) is shown as the dotted line; **this fit accurately predicts GPT-4’s final loss**. The x-axis is training compute normalized so that GPT-4 is 1.
    
    ![Untitled](GPT4%20629d4d85aaed4ef6a3817f0f6d7751e6/Untitled.png)
    
- **Multimodal Capabilities**: GPT-4 is distinguished by its ability to accept both image and text inputs, making it a significant advancement over its predecessors which were limited to text inputs.

- [**Emergent Abilities](https://zhuanlan.zhihu.com/p/609339534)** Large language models show emergent abilities at scale, where performance on a task remains at random levels until the model’s size reaches a certain threshold. After that, performance jumps and starts to improve as the model grows larger.

![Untitled](GPT4%20629d4d85aaed4ef6a3817f0f6d7751e6/Untitled%201.png)

- GPT4 performance

![Untitled](GPT4%20629d4d85aaed4ef6a3817f0f6d7751e6/Untitled%202.png)

- RLHF
    -  The report also discusses efforts to improve GPT-4's safety and alignment with human values through techniques like Reinforcement Learning from Human Feedback (RLHF), addressing concerns of bias, misinformation, and other ethical considerations in AI.

### **References**:

- [GPT4 research blog](https://openai.com/research/gpt-4)
- [Emergent Abilities](https://medium.com/@vdpappu/emergent-abilities-in-large-language-models-458e837e4a35)
- [Language Models are Few-Shot Learners](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&as_vis=1&q=Language+Models+are+Few-Shot+Learners&btnG=)
- [AndrejKarpathy GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)