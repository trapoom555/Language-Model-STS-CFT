# MTEB (Massive Text Embedding Benchmark)

In this project, our task focuses on evaluating on a STS (Semantic Textual Similarity) task in English that consist of the following benchmarks.

```python
TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark"
]
```

[MTEB github repository](https://github.com/embeddings-benchmark/mteb) provides us a convenient package for evaluation and is used in this evaluation script.

## Running Evaluation Script

### Define Benchmarks for Evaluations

All benchmarks for STS tasks are written in `minicpm_sts_eval.py`. If you want to test subset of the benchmarks, you can modify `TASK_LIST_STS` list.

### Model

The model for evaluation can be specified in `minicpm_sts_eval.py`. 

## Model class

There're 3 important methods in the class

1. `__init__(self, model_path='openbmb/MiniCPM-2B-dpo-bf16', adapter_path=None)` loading scripts for a model and a tokenizer are here. If the `adapter_path` is set to `None`, meaning that there will be no LoRA adapter applied.
2. `get_last_hidden_state(self, text)` is used for embedding vector extraction. The embedding vector at the last layer is used as a defult.
3. `encode(self, sentences, **kwargs)` giving a list of strings, this function return a list of embedding vectors in numpy array format.

### Run Evaluation

- For STS benchmarks

```bash
python minicpm_sts_eval.py
```

The evaluation results will be saved to `./results/minicpm/sts`.

## Compare with other models

The other model's evaluations can be seen in this [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) in the "STS" section.


