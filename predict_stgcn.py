"""
Inferencia con el modelo ST-GCN entrenado: carga stgcn_best.pt y predice
a qué palabra de LSP corresponde un skeleton dado.

Uso:
    python predict_stgcn.py "Dataset/Dataset/SUBJECT_1/Poor/sample_1/landmarks/skeleton_real.txt"

Si no se pasa una ruta por línea de comandos, se usa una ruta de ejemplo
definida en el bloque principal.
"""

import sys
import numpy as np
import torch

from preprocess_stgcn import process_sample
from stgcn_model import STGCN

DATA_DIR = "datos_procesados"
MODEL_DIR = "modelo"
CHECKPOINT_PATH = f"{MODEL_DIR}/stgcn_best.pt"
T_TARGET = 64  # debe coincidir con el T_TARGET usado en preprocesamiento/entrenamiento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    A = np.load(f"{DATA_DIR}/stgcn_adjacency.npy")
    classes = np.load(f"{DATA_DIR}/stgcn_label_classes.npy", allow_pickle=True)
    model = STGCN(num_classes=len(classes), A=A).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model, classes


def predict(path, model, classes, top_k=3):
    joints, missing_ratio = process_sample(path, T_target=T_TARGET)   # (C, T, V)
    x = torch.tensor(joints, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(-1)   # -> (1, C, T, V, 1) = batch de 1 muestra
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    resultados = [(classes[i], float(probs[i])) for i in top_idx]
    return resultados, missing_ratio


if __name__ == "__main__":
    muestra = sys.argv[1] if len(sys.argv) > 1 else \
        r"Dataset\Dataset\SUBJECT_1\Poor\sample_1\landmarks\skeleton_real.txt"

    print(f"Cargando modelo desde {CHECKPOINT_PATH}...")
    model, classes = load_model()

    print(f"\nMuestra: {muestra}")
    resultados, missing_ratio = predict(muestra, model, classes, top_k=3)

    print(f"% de landmarks imputados en esta muestra: {missing_ratio:.1%}")
    print("\nPredicciones (top 3):")
    for clase, prob in resultados:
        barra = "█" * int(prob * 30)
        print(f"  {clase:25s} {prob:6.1%}  {barra}")
