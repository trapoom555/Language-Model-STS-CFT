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

The other model's evaluations can be seen in this [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) in the Retrieval section.

## Running Evaluation Script

### Define Benchmarks for Evaluations

All benchmarks for retrieval task is written in a `minicpm_eval.py` file. If you want to test subset of the benchmarks, you can modify `TASK_LIST_RETRIEVAL` list.

### Model

The model for evaluation can be specified in `./model/minicpm.py`. There're 3 important methods in the class

1. `__init__()` loading a model and a tokenizer can be here.
2. `get_last_hidden_state()` is used for embedding vector extraction. The embedding vector at the last layer is used as a defult.
3. `encode()` giving a list of string, this function return a list of embedding vectors in numpy array format.

### Run Evaluation

```bash
python minicpm_eval.py
```

The results will be saved to `./results/minicpm`.
