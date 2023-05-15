import os

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Default config paths
EXPERIMENT_GROUPS_PATH = os.path.join(PROJECT_ROOT, "configs", "experiment_groups")
DMPAIR_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "dmpair_kwargs.yaml")
TRAINER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "trainer.yaml")
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "metrics.yaml")
ATTACK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "attack.yaml")
