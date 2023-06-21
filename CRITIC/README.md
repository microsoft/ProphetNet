# 🤔🛠️🤖 CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

This repository contains code and data for the paper "[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)".

## 💡 Introduction

**CRITIC** empowers LLMs to validate and rectify themselves through interaction with external tools.

<p align="center">
    <img src="./images/framework.png" width="1000">
</p>

> Humans typically utilize external tools to cross-check and reﬁne their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. 
> Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially “black boxes” to validate and progressively amend their own outputs in a manner similar to human interaction with tools.


## 💬 Examples

<p align="center">
    <img src="./images/demo.png" width="1000">
</p>


## 🛠️ Setup

We recommend the use of conda environments:

```sh
conda create --name critic python=3.8
conda activate critic
pip install -r requirements.txt
```

Configure APIs:

1. Configure the [LLMs API](https://platform.openai.com/docs/api-reference/introduction) in `src/llms/api.py`.

2. For truthfulness evaluation and fact correction, configure the [Google Search API](https://console.cloud.google.com/apis/api/customsearch.googleapis.com) in `src/tools/config.py`.

3. For toxicity reduction, you can follow this [tutorial](https://developers.google.com/codelabs/setup-perspective-api) and configure [Perspective API](https://www.perspectiveapi.com/) in `src/tools/config.py`.


## 🚀 Getting Started

We provide example bash scripts for each task as follows:

### Free-from Question Answering (Google)

- Inference: `bash scripts/run_qa_infer.sh`
- CRITIC: `bash scripts/run_qa_critic.sh`
- Evaluation: `python -m src.qa.evaluate`


### Mathematical Program Synthesis (Python Interpreter)

- Inference: `bash scripts/run_program_infer.sh`
- CRITIC: `bash scripts/run_program_critic.sh`
- Evaluation: `python -m src.program.evaluate`


### Toxicity Reduction (Perpective API)

- Inference: `bash scripts/run_toxicity_infer.sh`
- CRITIC: `bash scripts/run_toxicity_critic.sh`
- Evaluation: `python -m src.toxicity.evaluate`

## 🎯 Results

Example results with *gpt-3.5-turbo*:

Free-from Question Answering:

<p align="center">
    <img src="./images/qa_f1_iter_gpt-3.5-turbo.png" width="800">
</p>


Mathematical Program Synthesis:
<p align="center">
    <img src="./images/gsm8k_iter_gpt-3.5-turbo.png" width="250">
</p>

Toxicity Reduction:
<p align="center">
    <img src="./images/toxicity_iter_gpt-3.5-turbo.png" width="850">
</p>


## ❤️ Citation

```
@misc{gou2023critic,
      title={CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing}, 
      author={Zhibin Gou and Zhihong Shao and Yeyun Gong and Yelong Shen and Yujiu Yang and Nan Duan and Weizhu Chen},
      year={2023},
      eprint={2305.11738},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```