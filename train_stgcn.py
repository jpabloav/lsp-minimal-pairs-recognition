"""
Entrena el modelo ST-GCN sobre los datos preprocesados de LSP
(stgcn_X_train.npy, stgcn_y_train.npy, stgcn_adjacency.npy).

Guarda el mejor modelo (según F1-macro de validación) en stgcn_best.pt.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from stgcn_model import STGCN

# --- Configuración --------------------------------------------------------
DATA_DIR = "datos_procesados"
MODEL_DIR = "modelo"
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
VAL_SIZE = 0.15   # proporción del conjunto de entrenamiento reservada para validación
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"{MODEL_DIR}/stgcn_best.pt"
# ---------------------------------------------------------------------------


class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, criterion):
    """
    Devuelve (loss, accuracy, f1_macro). Con 34 clases desbalanceadas
    (ej. Green=277 vs Crowd=100), accuracy puede verse "bien" solo por
    acertar las clases mayoritarias. F1-macro promedia el F1 de CADA
    clase por igual, así que una clase minoritaria mal predicha sí
    penaliza la métrica -- es el criterio correcto para elegir el
    mejor modelo en un dataset desbalanceado como este.
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            total += X.size(0)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    accuracy = (all_preds == all_targets).mean()
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return total_loss / total, accuracy, f1_macro


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    print("Cargando datos preprocesados...")
    X_train_full = np.load(f"{DATA_DIR}/stgcn_X_train.npy")
    y_train_full = np.load(f"{DATA_DIR}/stgcn_Y_train.npy")
    A = np.load(f"{DATA_DIR}/stgcn_adjacency.npy")
    classes = np.load(f"{DATA_DIR}/stgcn_label_classes.npy", allow_pickle=True)
    num_classes = len(classes)

    # Nota: este split train/val es aleatorio (estratificado por clase),
    # no agrupado por sujeto, ya que el índice de sujeto por muestra de
    # train no se persiste en esta etapa. Sirve para monitorear el
    # entrenamiento; la evaluación rigurosa sigue siendo el test set,
    # que sí está separado por sujeto desde run_split_and_preprocess.py.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE,
        stratify=y_train_full, random_state=42,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Clases: {num_classes}")

    train_loader = DataLoader(SkeletonDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SkeletonDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = STGCN(num_classes=num_classes, A=A).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.2%} | val_f1_macro={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> nuevo mejor modelo guardado (val_f1_macro={val_f1:.4f}, val_acc={val_acc:.2%})")

    print(f"\nEntrenamiento terminado. Mejor val_f1_macro: {best_val_f1:.4f}")
    print(f"Checkpoint guardado en: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
