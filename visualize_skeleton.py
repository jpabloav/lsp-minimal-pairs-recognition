"""
Visualiza el skeleton (pose superior + 2 manos, 75 puntos) de una muestra,
para verificar visualmente si la mano se detecta bien en el video.

Genera:
  - Una imagen PNG por frame (útil para revisar frames específicos), o
  - Un GIF animado con todos los frames del clip (útil para ver el
    movimiento completo de la seña).

Usa las mismas conexiones de grafo que preprocess_stgcn.py, por lo que
la estructura mostrada corresponde exactamente al grafo que recibe el
modelo como entrada.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from preprocess_stgcn import (
    load_skeleton_real, _pose_edges, _hand_edges,
    LEFT_HAND_START, RIGHT_HAND_START, N_JOINTS,
)


def get_all_edges():
    edges = _pose_edges()
    edges += _hand_edges(LEFT_HAND_START)
    edges += _hand_edges(RIGHT_HAND_START)
    return edges


def plot_frame(ax, frame_xyz, edges, title=""):
    """frame_xyz: array (75, 3). Dibuja puntos y conexiones en 2D (x,y)."""
    ax.clear()
    x, y = frame_xyz[:, 0], frame_xyz[:, 1]

    # puntos válidos (no NaN) en azul, manos ausentes (NaN) no se dibujan
    valid = ~np.isnan(x) & ~np.isnan(y)
    ax.scatter(x[valid], y[valid], c="royalblue", s=12, zorder=3)

    # colorear cada mano distinto para verlas fácil
    left_hand = range(LEFT_HAND_START, LEFT_HAND_START + 21)
    right_hand = range(RIGHT_HAND_START, RIGHT_HAND_START + 21)
    ax.scatter(x[list(left_hand)], y[list(left_hand)], c="crimson", s=14, zorder=4, label="mano izq")
    ax.scatter(x[list(right_hand)], y[list(right_hand)], c="seagreen", s=14, zorder=4, label="mano der")

    for a, b in edges:
        if valid[a] and valid[b]:
            ax.plot([x[a], x[b]], [y[a], y[b]], c="gray", linewidth=1, zorder=2)

    ax.invert_yaxis()  # el eje y de MediaPipe crece hacia abajo, como en la imagen
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    left_missing = np.isnan(x[list(left_hand)]).all()
    right_missing = np.isnan(x[list(right_hand)]).all()
    estado = []
    if left_missing:
        estado.append("mano izq NO detectada")
    if right_missing:
        estado.append("mano der NO detectada")
    if estado:
        ax.text(0.02, 0.02, " | ".join(estado), transform=ax.transAxes,
                color="red", fontsize=9, va="bottom")


def save_single_frame(path, frame_idx=0, out_path="frame.png"):
    """Guarda una imagen PNG de un frame específico de la muestra."""
    joints = load_skeleton_real(path)  # (T, 75, 3)
    edges = get_all_edges()

    fig, ax = plt.subplots(figsize=(5, 7))
    plot_frame(ax, joints[frame_idx], edges, title=f"Frame {frame_idx}/{len(joints)-1}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def save_animation(path, out_path="skeleton.gif", fps=10):
    """Guarda un GIF animado recorriendo todos los frames de la muestra."""
    joints = load_skeleton_real(path)  # (T, 75, 3)
    edges = get_all_edges()
    T = joints.shape[0]

    fig, ax = plt.subplots(figsize=(5, 7))

    def update(frame_idx):
        plot_frame(ax, joints[frame_idx], edges, title=f"Frame {frame_idx}/{T-1}")

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    import os

    OUTPUT_DIR = "visualizaciones"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ruta a la muestra a visualizar
    MUESTRA = r"Dataset\SUBJECT_1\Poor\sample_1\landmarks\skeleton_real.txt"

    save_single_frame(MUESTRA, frame_idx=0, out_path=f"{OUTPUT_DIR}/frame_0.png")
    save_animation(MUESTRA, out_path=f"{OUTPUT_DIR}/skeleton.gif", fps=10)
