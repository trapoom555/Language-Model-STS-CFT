# Language-Model-STS-CFT

[Paper Coming Soon...] [[Hugging Face 🤗](https://huggingface.co/collections/trapoom555/small-lms-text-embedding-663b3ec87527788a577f6852)]

This project aims to improve text embedding of smaller Language Models (LMs) up to 2B parameters using the contrastive fine-tuning technique. Specifically, the InfoNCE loss is utilized as a training objective.

$$\min  - \log \frac{e^{\text{sim}(\textbf{h}_i, \textbf{h}_i^+) / \tau}}{\sum_i \left( e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^+) / \tau }+ e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^-) / \tau} \right)}$$

where $\textbf{h}_i$ denotes an embedding vector of a premise $x_i$, $\tau$ denotes a temperature and $\text{sim}(\textbf{h}_i, \textbf{h}_i^+)$ computes the cosine similarity between embedding vectors $\textbf{h}_i$ and $\textbf{h}_i^+$.

We employ LoRA as our parameter-efficient fine-tuning technique in order to reduce the memory requirement.

## Embedding Extraction

- Every prompt will be appended by the [EOS] token.
- The embedding vector will be extracted from hidden states at the last layer of this [EOS] token.

## Fine-tuned Weights

We have fine-tuned 3 models and we provide their LoRA adapter weights in [this Hugging Face 🤗 collection](https://huggingface.co/collections/trapoom555/small-lms-text-embedding-663b3ec87527788a577f6852). 

The base models consist of
1. MiniCPM-2B-dpo-bf16
2. Gemma-2B-it
3. Phi-2

The performance and fine-tuning details can be seen in the Hugging Face model page.

## Dataset

We utilize the processed NLI dataset as our fine-tuning dataset. The dataset consists of 275K triplets of anchors, their corresponding entailments along with hard negatives. Please follow [this README](https://github.com/trapoom555/Language-Model-STS-CFT/blob/main/data/README.md) to see how to download the dataset.

## Fine-tuning with your own resources

If you are willing to fine-tune the LMs with your own resources, we've provided the code for you. Our code can work with multi-GPUs settings. The more GPUs you have, the larger batch size you can fine-tune.

First, you need to setup the virtual environment. We provided the environment setup file you you.

```bash
conda env create --file environment.yml
conda activate cft
```
Then, download the processed NLI dataset following [this README](https://github.com/trapoom555/Language-Model-STS-CFT/blob/main/data/README.md)

After that, please follow [this README](https://github.com/trapoom555/Language-Model-STS-CFT/blob/main/train/README.md) for the fine-tuning steps.

## Footnote

This work is the final project of the Natural Language Processing Spring 2024 course at Tsinghua University 🟣. We would like to express our sincere gratitude to this course !
