## Reasoning over Uncertain Text by Generative Large Language Models

Presented at: AAAI

URL: https://arxiv.org/abs/2402.09614

Abstract:

This paper considers the challenges that Large Language Models (LLMs) face when reasoning over text that includes information involving uncertainty explicitly quantified via probability values. This type of reasoning is relevant to a variety of contexts ranging from everyday conversations to medical decision-making. Despite improvements in the mathematical reasoning capabilities of LLMs, they still exhibit significant difficulties when it comes to probabilistic reasoning. To deal with this problem, we first introduce the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs. We then leverage this new dataset to thoroughly illustrate the specific limitations of LLMs for tasks involving probabilistic reasoning and present several strategies that map the problem to different formal representations, including Python code, probabilistic inference algorithms, and probabilistic logical programming. We conclude by providing an evaluation of our methods on BLInD and on an adaptation of a causal reasoning question-answering dataset, which further shows their practical effectiveness.


## What Are the Odds? Language Models Are Capable of Probabilistic Reasoning

Presented at: ACL

URL: https://arxiv.org/abs/2406.12830

Abstract:

Language models (LM) are capable of remarkably complex linguistic tasks; however, numerical reasoning is an area in which they frequently struggle. An important but rarely evaluated form of reasoning is understanding probability distributions. In this paper, we focus on evaluating the probabilistic reasoning capabilities of LMs using idealized and real-world statistical distributions. We perform a systematic evaluation of state-of-the-art LMs on three tasks: estimating percentiles, drawing samples, and calculating probabilities. We evaluate three ways to provide context to LMs 1) anchoring examples from within a distribution or family of distributions, 2) real-world context, 3) summary statistics on which to base a Normal approximation. Models can make inferences about distributions, and can be further aided by the incorporation of real-world context, example shots and simplified assumptions, even if these assumptions are incorrect or misspecified. To conduct this work, we developed a comprehensive benchmark distribution dataset with associated question-answer pairs that we have released publicly. 
