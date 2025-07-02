from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PlotColors:
    CQL: str = "#1f77b4"
    IQL: str = "#ff7f0e"
    TD3_BC: str = "#2ca02c"

@dataclass
class DashboardConfig:
    db_path: str = "db/experiments.db"
    available_envs: List[str] = field(default_factory=lambda: [
        "hopper-medium-v2",
        "halfcheetah-medium-expert-v2"
    ])
    plot_colors: Dict[str, str] = field(default_factory=lambda: PlotColors().__dict__)
    enable_caching: bool = True
    default_seed: int = 0
