"""2D top-down room visualization panel for Streamlit dashboard."""

from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from rf_sim.scenario import ScenarioConfig

MATERIAL_COLORS = {
    "drywall": "#95a5a6",
    "wood": "#d35400",
    "brick": "#c0392b",
    "concrete": "#34495e",
}


def plot_room_view(
    config: ScenarioConfig,
    human_pos: Tuple[float, float] = None,
) -> plt.Figure:
    """Generates a 2D top-down Matplotlib figure of the room layout, walls, nodes, and human.

    Args:
        config: ScenarioConfig instance specifying room, walls, and node locations.
        human_pos: Optional (x, y) tuple of current human coordinates.

    Returns:
        plt.Figure: Matplotlib figure instance.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # 1. Room Outer Bounds
    ax.set_xlim(-0.5, config.room_width + 0.5)
    ax.set_ylim(-0.5, config.room_height + 0.5)
    room_rect = patches.Rectangle(
        (0, 0),
        config.room_width,
        config.room_height,
        linewidth=2,
        edgecolor="#2c3e50",
        facecolor="#ecf0f1",
        alpha=0.5,
    )
    ax.add_patch(room_rect)

    # 2. Draw Walls
    for wall in config.walls:
        (x1, y1), (x2, y2) = wall.start_point, wall.end_point
        color = MATERIAL_COLORS.get(wall.material.lower(), "#7f8c8d")
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=max(3.0, wall.thickness_m * 20),
            solid_capstyle="round",
            label=f"Wall ({wall.material})",
        )

    # 3. Draw TX and RX Nodes
    ax.plot(config.tx_x, config.tx_y, "^", color="#2980b9", markersize=10, label="TX Node")
    ax.annotate("TX", (config.tx_x, config.tx_y + 0.3), fontsize=9, fontweight="bold", ha="center", color="#1a5276")

    ax.plot(config.rx_x, config.rx_y, "s", color="#27ae60", markersize=10, label="RX Node")
    ax.annotate("RX", (config.rx_x, config.rx_y + 0.3), fontsize=9, fontweight="bold", ha="center", color="#1e8449")

    # 4. Draw Human Target if present and position provided
    if config.person_present and human_pos is not None:
        hx, hy = human_pos
        ax.plot(hx, hy, "o", color="#e74c3c", markersize=12, label="Human Target")
        ax.plot(hx, hy, "+", color="#ffffff", markersize=8, markeredgewidth=2)
        ax.annotate("Target", (hx, hy + 0.35), fontsize=9, fontweight="bold", ha="center", color="#922b21")

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_title(f"Room Top-Down View ({config.room_width:.1f}m × {config.room_height:.1f}m)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("X Position (m)", fontsize=9)
    ax.set_ylabel("Y Position (m)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig
