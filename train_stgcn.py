"""
Entrena el modelo ST-GCN sobre los datos preprocesados de LSP
(stgcn_X_train.npy, stgcn_y_train.npy, stgcn_adjacency.npy).

Guarda el mejor modelo (según accuracy de validación) en stgcn_best.pt.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from stgcn_model import STGCN

# --- Configuración: ajusta si hace falta -------------------------------
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
VAL_SIZE = 0.15   # % de tu train actual que se reserva para validación
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "stgcn_best.pt"
# -------------------------------------------------------------------------


class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def main():
    print(f"Dispositivo: {DEVICE}")
    print("Cargando datos preprocesados...")
    X_train_full = np.load("stgcn_X_train.npy")
    y_train_full = np.load("stgcn_y_train.npy")
    A = np.load("stgcn_adjacency.npy")
    classes = np.load("stgcn_label_classes.npy", allow_pickle=True)
    num_classes = len(classes)

    # Nota: este split train/val es aleatorio (estratificado por clase),
    # NO agrupado por sujeto, porque no guardamos a qué sujeto pertenece
    # cada muestra de train. Sirve para monitorear el entrenamiento; la
    # evaluación real y rigurosa sigue siendo el test set (ese sí está
    # separado por sujeto desde run_split_and_preprocess.py).
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

    best_val_acc = 0.0
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
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> nuevo mejor modelo guardado ({val_acc:.2%})")

    print(f"\nEntrenamiento terminado. Mejor val_acc: {best_val_acc:.2%}")
    print(f"Checkpoint guardado en: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()