import os

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Configs
DATASET_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "datasets.yaml")
TRAINER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "trainer.yaml")
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "metrics.yaml")
