from __future__ import annotations

import sys
from pathlib import Path
import shutil

# Ensure repository root is on sys.path so we can import config and graphs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx

from config import ControlsConfig, UDGConfig, ProjectConfig, load_project_config_json
from graphs.unit_disk import generate_square_lattice_udg


def sample_udg_graphs(
    project: ProjectConfig,
    n_samples: int = 4,
    base_seed: int = 0,
) -> List[Tuple[nx.Graph, Dict[int, tuple[float, float]], int]]:
    """Generate several defective square lattice UDG samples.

    Parameters
    ----------
    project : ProjectConfig
        Shared project configuration (controls and UDG settings).
    n_samples : int
        Number of graphs to sample.
    base_seed : int
        Base seed; each sample uses base_seed + i for deterministic variety.

    Returns
    -------
    list of (G, pos, seed)
        Graphs with positions and the seed used for generation.
    """
    out: List[Tuple[nx.Graph, Dict[int, tuple[float, float]], int]] = []
    for i in range(n_samples):
        seed = project.udg.seed + base_seed + i
        udg = UDGConfig(
            nx=project.udg.nx,
            ny=project.udg.ny,
            spacing=project.udg.spacing,
            radius=project.udg.radius,
            dropout_rate=project.udg.dropout_rate,
            seed=seed,
        )
        G, pos = generate_square_lattice_udg(udg)
        out.append((G, pos, seed))
    return out


essential_style = dict(node_size=20, width=1.0, edge_color="#888", node_color="#1f77b4")


def visualize_samples(
    samples: List[Tuple[nx.Graph, Dict[int, tuple[float, float]], int]],
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Path | None:
    """Visualize a list of UDG samples in a grid of subplots and optionally save.

    If `save_path` is provided, the figure is saved there and not shown.
    Returns the saved path if provided, else None.
    """
    n = len(samples)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for idx, (G, pos, seed) in enumerate(samples):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.set_title(f"seed={seed} |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=False,
            **essential_style,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def main() -> None:
    # Load experiment-wide configuration from config.json at repo root
    config_path = ROOT / "config.json"
    project = load_project_config_json(config_path)
    udg = project.udg
    print(
        f"Config: nx={udg.nx}, ny={udg.ny}, radius={udg.radius}, "
        f"dropout={udg.dropout_rate}, seed={udg.seed}"
    )

    n_samples = 4
    base_seed = 0
    samples = sample_udg_graphs(project, n_samples=n_samples, base_seed=base_seed)

    # Save under visualization/figures
    out_dir = Path(__file__).resolve().parent / "figures"
    fname = (
        f"udg_samples_nx{udg.nx}_ny{udg.ny}_r{udg.radius:.2f}_drop{udg.dropout_rate:.2f}"
        f"_base{base_seed}_n{n_samples}.png"
    )
    save_path = out_dir / fname
    result_path = visualize_samples(
        samples, title="Defective square-lattice UDG samples", save_path=save_path
    )
    if result_path is not None:
        # Also write/update a stable filename for convenience
        latest_path = out_dir / "udg_samples_latest.png"
        try:
            shutil.copyfile(result_path, latest_path)
        except Exception:
            pass
        print(f"Loaded config from: {config_path}")
        print(f"Saved figure to: {result_path}")



if __name__ == "__main__":
    main()
