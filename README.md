# Language-Model-STS-CFT

This project aims to improve text embedding of LLM using the contrastive fine-tuning technique. Specifically, the InfoNCE loss is utilized as a training objective.

$$\min  - \log \frac{e^{\text{sim}(\textbf{h}_i, \textbf{h}_i^+) / \tau}}{\sum_i \left( e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^+) / \tau }+ e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^-) / \tau} \right)}$$

## Footnote

This work is the final project of the Natural Language Processing Spring 2024 course at Tsinghua University 🟣. We would like to express our sincere gratitude to this course !
