from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PBTConfig:
    # Population Settings
    population_size: int = 4
    generations: int = 10

    # Training
    episodes_per_gen: int = 100
    mcts_sims: int = 800
    batch_size: int = 64
    epochs_per_gen: int = 1

    # Evolution
    mutation_rate: float = 0.2

    # Paths
    project_root: str = "."
    data_dir: str = "data/pbt_data"
    model_dir: str = "models/pbt_models"
    meta_deck_path: str = "data/meta_decks.json"

    # Neural Network
    model_type: str = "transformer"  # or 'resnet'

    def __post_init__(self):
        import os
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
