# Randomness control:
from pathlib import Path
import torch
import numpy as np


global_seed = 42
#torch.manual_seed(global_seed)
rng = np.random.default_rng()

# Pytorch device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convenience imports:
from .genetic_algorithm.evolution import Evolution  # noqa
from .model.BNCmodel import BNCmodel  # noqa
from .plot.pareto import generate_pareto_animation  # noqa
from .plot.pareto import Benchmark  # noqa
