+++
title = "large language models are not intelligent."
date = 2024-04-30
description = "🌳"
+++

I've found myself using LLMs less and less over time. They're great for generating summaries, rephrasing writing, explaining concepts, and writing some code—things I refer to as "manual labor." But if you ask them to do anything reasoning-heavy, they fail very often.

Today's LLMs are not actually intelligent; they're just really good stochastic parrots.

### Memorization is a big deal

About 4 months ago, I read a paper titled [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/abs/2311.17035). That was when I realized memorization is a *powerful* tool for LLM performance.

> Aside (for context): DeepMind previously found [laws](https://arxiv.org/abs/2203.15556) governing the optimal model size/number of training tokens under a fixed FLOPS budget. Popular models, such as LLaMA-7B and Mistral-7B may be trained with few parameters—but they're overtrained compared to these Chinchilla-optimal token counts. LLaMA3, for instance, was pretrained on over [15 trillion tokens](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).

When attacking both closed LLMs (like GPT-3.5-instruct) and semi-closed models (such as LLaMA), the authors found a significant relationship between overtraining and memorization. Nearly 0.30% of LLaMA-7B's generated tokens were memorized. And nearly 0.80% of LLaMA-65B's generated tokens were memorized! Meanwhile, around 0.50% of Mistral-7B's generated tokens were memorized.

*And these are lower bounds!*

On the other hand, undertrained models such as OPT-1.3B and OPT-6.7B had much lower memorization rates (around 0.03% and 0.09%, respectively).

The percentages may seem small, but the results indicate that language models regularly spit out memorized training data. And memorization rates of 0.30%-0.80% in the overtrained models are nontrivial.

### What do benchmarks even measure?

It's probably not a coincidence that these overtrained, memorizing models are exactly the ones that perform well on popular benchmarks. For instance, LLaMA-65B and Mistral-7B score significantly [higher](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) on the MMLU benchmark compared to the undertrained OPT-66B.

The [TinyLlama](https://github.com/jzhang38/TinyLlama) project provides some more evidence for this hypothesis. They overtrained a 1.1-billion parameter model with 3 trillion tokens. All else held constant, the Commonsense benchmark scores increase approximately linearly with the number of training tokens.

This relationship between overtraining and benchmarks leads me to wonder: What do benchmarks even measure? We use benchmark scores as a proxy for knowledge, reasoning, and intelligence. But are they really measuring those abilities? Or are they measuring memorization? Memorization isn't intelligence—and if the pretraining dataset may be [contaminated](https://arxiv.org/pdf/2311.04850) by benchmark data, these benchmark scores tell us even less about the intelligence of our LLMs.

### It's all about the data

About a week ago, I read a little post titled [The “it” in AI models is the dataset](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/). And that got me thinking...there's a reason Meta, Mistral, and other companies that release "open" models don't release their training data.

[This paper](https://arxiv.org/abs/2309.14402) from Meta uses controlled studies to really show how much training data matters.  Successfully fine-tuning a LLM on question-answering tasks depends *highly* on data augmentation: One of the paper's experiments shows a 9.7% knowledge extraction accuracy without augmentation, but a 96.6% accuracy with data augmentation.

Why? If the model hasn't seen a question phrased in a particular way during training, it'll [probably perform worse](https://arxiv.org/abs/2306.11270) on that question during generation. Augmentation gives the LLM a higher diversity of data, so that no matter how a user phrases a question, the model is more likely to have seen that phrasing before. (In other words, in-distribution performance is much better than out-of-distribution performance.)

Again—this research suggests memorization, not intelligence.

### Limitations on Reasoning

I've repeatedly mentioned that memorization and intelligence are not the same. But is that really the case? The answer is yes: Even if an LLM "memorizes" a fact, it cannot reliably answer basic *reasoning* questions about that fact.

If a model sees the sentence, "BTS's debut song is titled 'No More Dream,'" it likely won't be able to answer the question, "What artist debuted with the song 'No More Dream'?" LLMs struggle with this kind of task, called *inverse search*.

And LLMs cannot reliably answer comparison questions, either. From the Meta [paper](https://arxiv.org/abs/2309.14402):
> Determining whether a month is even or odd requires 10,000 training samples to achieve a 75% accuracy, despite theoretically needing a sample complexity on the order of O(12).
> 
> Ranking months \[such as indicating that April comes before July\] requires 50,000 training samples to reach an 85% test accuracy, even with a theoretical sample complexity of O(122), provided no hint is given.

Surprisingly, <u>this holds regardless of finetuning, prompting, or pretraining.</u>

All this is bad news if we expect today's LLMs to genuinely become intelligent.

But if you're just trying to quickly write some code to generate Matplotlib figures, things have never been better.

