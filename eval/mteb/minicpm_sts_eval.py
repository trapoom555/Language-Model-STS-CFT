from model.minicpm import MiniCPM
from mteb import MTEB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

model_path = '../../pretrained/MiniCPM-2B-dpo-bf16'
adapter_path = '../../pretrained/adapter/20240422020420'

model = MiniCPM(model_path=model_path,
                adapter_path=adapter_path)

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

for task in TASK_LIST_STS:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(tasks=[task], task_langs=["en"])
    evaluation.run(model, output_folder=f"results/minicpm/sts")


