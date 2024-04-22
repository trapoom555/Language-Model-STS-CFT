# MTEB (Massive Text Embedding Benchmark)

In this project, our task focuses on evaluating on a retrieval task in English that consists of the following benchmarks.

```python
TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID"
]
```

[MTEB github repository](https://github.com/embeddings-benchmark/mteb) provides us a convenient package for evaluation and is used in this evaluation script.

## Running Evaluation Script

### Define Benchmarks for Evaluations

All benchmarks for retrieval task is written in a `minicpm_eval.py` file. If you want to test subset of the benchmarks, you can modify `TASK_LIST_RETRIEVAL` list.

### Model

The model for evaluation can be specified in `./model/minicpm.py`. There're 3 important methods in the class

1. `__init__(self)` loading script for a model and a tokenizer can be here.
2. `get_last_hidden_state(self, text)` is used for embedding vector extraction. The embedding vector at the last layer is used as a defult.
3. `encode(self, sentences, **kwargs)` giving a list of strings, this function return a list of embedding vectors in numpy array format.

### Run Evaluation

- For STS benchmarks

```bash
python minicpm_sts_eval.py
```

- For Retrieval benchmarks

```bash
python minicpm_retrieval_eval.py
```

The evaluation results will be saved to `./results/minicpm/sts` and `./results/minicpm/retrieval` respectively.

## Compare with other models

The other model's evaluations can be seen in this [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) in a "Retrieval" section.


