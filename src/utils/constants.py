from pathlib import Path

HOME_DIR = str(Path(__file__).resolve().parents[2])

DATA_DIR = f"{HOME_DIR}/data"
MNIST_DIR = f"{DATA_DIR}/mnist"
SUPERVISED_DIR = f"{MNIST_DIR}/supervised"
WEAKLY_SUPERVISED_DIR = f"{MNIST_DIR}/weakly_supervised"

TRAINING_DIR = f"{HOME_DIR}/training_results"
SUPERVISED_TRAINING_DIR = f"{TRAINING_DIR}/supervised"
SIAMESE_TRAINING_DIR = f"{TRAINING_DIR}/siamese"
WEAKLY_SUPERVISED_TRAINING_DIR = f"{TRAINING_DIR}/weakly_supervised"
SCALAR_RANKING_TRAINING_DIR = f"{TRAINING_DIR}/scalar_ranking"
JOINT_RELATIONAL_TRAINING_DIR = f"{TRAINING_DIR}/joint_relational"




