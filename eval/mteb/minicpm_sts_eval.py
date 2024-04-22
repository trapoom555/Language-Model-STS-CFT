from model.minicpm import MiniCPM
from mteb import MTEB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

model = MiniCPM()

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
    "STSBenchmark",
    "CDSC-R"
]

for task in TASK_LIST_STS:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(tasks=[task], task_langs=["en"])
    evaluation.run(model, output_folder=f"results/minicpm/sts")


