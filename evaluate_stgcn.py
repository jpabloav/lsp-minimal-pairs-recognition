"""
Evaluación del modelo ST-GCN entrenado sobre el conjunto de prueba
(sujetos no vistos durante el entrenamiento).

Entradas (generadas por run_split_and_preprocess.py y train_stgcn.py):
    stgcn_X_test.npy, stgcn_y_test.npy, stgcn_adjacency.npy,
    stgcn_label_classes.npy, stgcn_best.pt

Salidas:
    - Métricas globales (accuracy, F1-macro, F1-weighted) por consola
    - Reporte por clase (precision, recall, F1) por consola y en CSV
    - Matriz de confusión normalizada por fila (PNG)
    - Análisis por grupos de minimal pairs (consola)

Uso:
    python evaluate_stgcn.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana, apto para servidores/Colab
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA_DIR = "datos_procesados"
MODEL_DIR = "modelo"
RESULTS_DIR = "resultados"
CHECKPOINT_PATH = f"{MODEL_DIR}/stgcn_best.pt"
BATCH_SIZE = 64

# Grupos de clases que se comparan entre sí como minimal pairs (o tríadas).
# Los nombres deben coincidir con los de stgcn_label_classes.npy. La
# agrupación es una propuesta basada en los nombres de las clases y debe
# validarse contra la definición de minimal pairs del dataset.
MINIMAL_PAIR_GROUPS = [
    ("Poor", "Very_poor"),
    ("Tall", "Very_tall", "Very_short"),
    ("Small_cell_phone", "Large_cell_phone", "Extra_large_cell_phone"),
    ("Learn_something", "Learn_everything"),
    ("Person", "People"),
]


def load_model_and_data():
    import torch
    from stgcn_model import STGCN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_test = np.load(f"{DATA_DIR}/stgcn_X_test.npy")
    y_test = np.load(f"{DATA_DIR}/stgcn_Y_test.npy")
    A = np.load(f"{DATA_DIR}/stgcn_adjacency.npy")
    classes = np.load(f"{DATA_DIR}/stgcn_label_classes.npy", allow_pickle=True)

    model = STGCN(num_classes=len(classes), A=A).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model, device, X_test, y_test, classes


def predict_all(model, device, X, batch_size=BATCH_SIZE):
    import torch

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def global_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print("=== Métricas globales (test) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-macro:    {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    return acc, f1_macro, f1_weighted


def per_class_report(y_true, y_pred, classes, csv_path=None):
    if csv_path is None:
        csv_path = f"{RESULTS_DIR}/stgcn_per_class_metrics.csv"
    labels = np.arange(len(classes))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(classes),
        zero_division=0, output_dict=True,
    )
    print("\n=== Reporte por clase (test) ===")
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=list(classes), zero_division=0,
    ))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("clase,precision,recall,f1,soporte\n")
        for c in classes:
            r = report[c]
            f.write(f"{c},{r['precision']:.4f},{r['recall']:.4f},"
                    f"{r['f1-score']:.4f},{int(r['support'])}\n")
    print(f"Métricas por clase guardadas en {csv_path}")


def plot_confusion_matrix(y_true, y_pred, classes, out_path=None):
    if out_path is None:
        out_path = f"{RESULTS_DIR}/stgcn_confusion_matrix.png"
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión normalizada por fila (test)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusión guardada en {out_path}")


def minimal_pair_analysis(y_true, y_pred, classes, groups=MINIMAL_PAIR_GROUPS):
    """
    Para cada grupo de minimal pairs reporta, sobre las muestras cuya clase
    real pertenece al grupo: accuracy dentro del grupo, y cuántos errores
    fueron confusiones con otra clase del mismo grupo frente a errores hacia
    clases externas.
    """
    name_to_idx = {c: i for i, c in enumerate(classes)}
    print("\n=== Análisis por grupos de minimal pairs (test) ===")
    for group in groups:
        missing = [g for g in group if g not in name_to_idx]
        if missing:
            print(f"\nGrupo {group}: omitido, clases no encontradas: {missing}")
            continue

        idxs = [name_to_idx[g] for g in group]
        mask = np.isin(y_true, idxs)
        n = int(mask.sum())
        if n == 0:
            print(f"\nGrupo {group}: sin muestras en test")
            continue

        yt, yp = y_true[mask], y_pred[mask]
        correct = int((yt == yp).sum())
        within = int(((yt != yp) & np.isin(yp, idxs)).sum())
        outside = int(((yt != yp) & ~np.isin(yp, idxs)).sum())

        print(f"\nGrupo {group}  ({n} muestras)")
        print(f"  Acierto dentro del grupo:      {correct}/{n} ({correct / n:.1%})")
        print(f"  Errores hacia otra clase del grupo: {within}")
        print(f"  Errores hacia clases externas:      {outside}")
        for a in group:
            ia = name_to_idx[a]
            sub = yt == ia
            if sub.sum() == 0:
                continue
            row = ", ".join(
                f"{classes[j]}={int((yp[sub] == j).sum())}" for j in idxs
            )
            print(f"    real {a} ({int(sub.sum())}) -> {row}")


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, device, X_test, y_test, classes = load_model_and_data()
    print(f"Muestras de test: {len(y_test)}  |  Clases: {len(classes)}  |  Dispositivo: {device}\n")

    y_pred = predict_all(model, device, X_test)

    global_metrics(y_test, y_pred)
    per_class_report(y_test, y_pred, classes)
    plot_confusion_matrix(y_test, y_pred, classes)
    minimal_pair_analysis(y_test, y_pred, classes)


if __name__ == "__main__":
    main()
