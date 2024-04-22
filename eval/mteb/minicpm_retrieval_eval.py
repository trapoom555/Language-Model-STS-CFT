from model.minicpm import MiniCPM
from mteb import MTEB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

model_path = '../../pretrained/MiniCPM-2B-dpo-bf16'
adapter_path = '../../pretrained/adapter/20240422020420'

model = MiniCPM(model_path=model_path,
                adapter_path=adapter_path)

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
    "TRECCOVID",
]

for task in TASK_LIST_RETRIEVAL:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(tasks=[task], task_langs=["en"])
    evaluation.run(model, output_folder=f"results/minicpm/retrieval")


