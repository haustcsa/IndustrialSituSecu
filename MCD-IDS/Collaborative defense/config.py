
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


class GlobalConfig:

    EXPERIMENT_TIME = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    ORIGINAL_RAW_DATA_PATH = str(
        PROJECT_ROOT / "datasets" / "CIC_IDS_2017" / "MachineLearningCSV" / "MachineLearningCVE"
    )
    DATA_FILES = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]


    PROCESSED_DATA_PATH = str(PROJECT_ROOT / "datasets" / "CIC_IDS_2017" / "processed_data")

    INITIAL_MODEL_SAVE_DIR = str(PROJECT_ROOT / "models" / "initial")

    LOG_BASE_DIR = str(PROJECT_ROOT / "logs")

    HDF_KEY = "cic_ids_2017"
    RANDOM_SEED = 42

    INPUT_DIM = 70
    NUM_TRAINING_CLASSES = 3
    TARGET_NAMES = ["Benign", "Known Attack", "Unknown Attack"]

    Z_DIM = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    CENTER_LOSS_WEIGHT = 0.0
    EPOCHS = 10
    NUM_NODES = 5

    TRUST_INITIAL = 1.0
    TRUST_EMA_ALPHA = 0.1
    TRUST_REWARD = 0.05
    TRUST_PENALTY = 0.05
    MIN_TRUST = 0.01
    MAX_TRUST = 10.0

    USE_BAYESIAN_FUSION = True
    USE_CLASS_PRIORS = True
    CLS_PRIORS = {"Benign": 0.7, "Known Attack": 0.25, "Unknown Attack": 0.05}

    POSTERIOR_MIN_CONF = 0.5
    EPSILON = 1e-12

    NODE_QUEUE_TIMEOUT = 0.1
    COORDINATOR_QUEUE_TIMEOUT = 0.1
    MANAGER_COORDINATOR_TIMEOUT_S = 120



config = GlobalConfig()
