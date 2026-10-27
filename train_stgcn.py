"""
Entrenamiento del modelo ST-GCN sobre los datos preprocesados de LSP.

Lee los tensores generados por run_split_and_preprocess.py para un T dado
(<datos>/T<T>/) y usa la validación por sujeto definida en la partición
(stgcn_X_val.npy, stgcn_Y_val.npy). El mejor modelo según F1-macro de
validación se guarda en <modelo-dir>/<arq>_T<T>_s<semilla>.pt, junto con el
historial por época (CSV) y un resumen de la corrida (JSON).

Protocolo: Adam, parada temprana por F1-macro de validación, semilla fija.

Uso:
    python train_stgcn.py --T 32
    python train_stgcn.py --T 32 --epochs 2      # prueba rápida
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from modelos import (
    ARQUITECTURAS, ParadaTemprana, crear_modelo, nombre_corrida, ruta_checkpoint,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrenamiento ST-GCN para LSP.")
    p.add_argument("--T", type=int, default=32, help="frames por muestra (debe existir <datos>/T<T>/)")
    p.add_argument("--arq", default="simplificado", choices=ARQUITECTURAS, help="arquitectura a entrenar")
    p.add_argument("--semilla", type=int, default=42, help="semilla de reproducibilidad")
    p.add_argument("--epochs", type=int, default=50, help="máximo de épocas")
    p.add_argument("--batch", type=int, default=32, help="tamaño de batch")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate de Adam")
    p.add_argument("--paciencia", type=int, default=10, help="épocas sin mejora de F1-macro antes de detener")
    p.add_argument("--datos", default="datos_procesados", help="carpeta base de los datos preprocesados")
    p.add_argument("--modelo-dir", default="modelo", help="carpeta de salida de modelos e historiales")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def cargar_datos(datos_dir, T):
    """Carga train, val, adyacencia y clases de <datos_dir>/T<T>/."""
    d = os.path.join(datos_dir, f"T{T}")
    archivos = [
        "stgcn_X_train.npy", "stgcn_Y_train.npy", "stgcn_X_val.npy", "stgcn_Y_val.npy",
        "stgcn_adjacency.npy", "stgcn_label_classes.npy",
    ]
    faltantes = [a for a in archivos if not os.path.exists(os.path.join(d, a))]
    if faltantes:
        raise SystemExit(
            f"Faltan archivos en {d}: {faltantes}. "
            f"Generarlos con: python run_split_and_preprocess.py --T {T}"
        )
    X_train = np.load(os.path.join(d, "stgcn_X_train.npy"))
    Y_train = np.load(os.path.join(d, "stgcn_Y_train.npy"))
    X_val = np.load(os.path.join(d, "stgcn_X_val.npy"))
    Y_val = np.load(os.path.join(d, "stgcn_Y_val.npy"))
    A = np.load(os.path.join(d, "stgcn_adjacency.npy"))
    classes = np.load(os.path.join(d, "stgcn_label_classes.npy"), allow_pickle=True)
    return X_train, Y_train, X_val, Y_val, A, classes


def evaluate(model, loader, criterion):
    """
    Devuelve (loss, accuracy, f1_macro). Con 34 clases desbalanceadas,
    accuracy puede verse alto solo por acertar las clases mayoritarias.
    F1-macro promedia el F1 de cada clase por igual, por lo que una clase
    minoritaria mal predicha sí penaliza la métrica.
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


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.semilla)
    os.makedirs(args.modelo_dir, exist_ok=True)

    nombre = nombre_corrida(args.arq, args.T, args.semilla)
    ckpt_path = ruta_checkpoint(args.arq, args.T, args.semilla, args.modelo_dir)
    hist_path = os.path.join(args.modelo_dir, f"{nombre}_historial.csv")
    resumen_path = os.path.join(args.modelo_dir, f"{nombre}_resumen.json")

    print(f"Corrida: {nombre}  |  Dispositivo: {DEVICE}")
    X_train, Y_train, X_val, Y_val, A, classes = cargar_datos(args.datos, args.T)
    num_classes = len(classes)
    print(f"Train: {len(Y_train)}  Val: {len(Y_val)}  Clases: {num_classes}  "
          f"Forma de una muestra: {X_train.shape[1:]}")

    train_loader = DataLoader(
        SkeletonDataset(X_train, Y_train), batch_size=args.batch, shuffle=True,
        generator=torch.Generator().manual_seed(args.semilla),
    )
    val_loader = DataLoader(SkeletonDataset(X_val, Y_val), batch_size=args.batch)

    model = crear_modelo(args.arq, num_classes, A).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    parada = ParadaTemprana(args.paciencia)

    mejor_epoca = 0
    epocas_ejecutadas = 0
    t0 = time.time()
    with open(hist_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoca", "train_loss", "val_loss", "val_acc", "val_f1_macro"])

        for epoch in range(1, args.epochs + 1):
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
            epocas_ejecutadas = epoch

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_acc:.6f}", f"{val_f1:.6f}"])
            fh.flush()
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | val_acc={val_acc:.2%} | val_f1_macro={val_f1:.4f}")

            mejoro, detener = parada.actualizar(val_f1)
            if mejoro:
                mejor_epoca = epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> nuevo mejor modelo guardado (val_f1_macro={val_f1:.4f})")
            if detener:
                print(f"Parada temprana: {args.paciencia} épocas sin mejora de F1-macro.")
                break

    resumen = {
        "corrida": nombre,
        "arquitectura": args.arq,
        "T": args.T,
        "semilla": args.semilla,
        "hiperparametros": {"lr": args.lr, "batch": args.batch, "epochs_max": args.epochs,
                            "paciencia": args.paciencia},
        "muestras": {"train": int(len(Y_train)), "val": int(len(Y_val))},
        "epocas_ejecutadas": epocas_ejecutadas,
        "mejor_epoca": mejor_epoca,
        "mejor_val_f1_macro": parada.mejor,
        "segundos": round(time.time() - t0, 1),
        "dispositivo": DEVICE,
    }
    with open(resumen_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nEntrenamiento terminado. Mejor val_f1_macro: {parada.mejor:.4f} (época {mejor_epoca})")
    print(f"Checkpoint: {ckpt_path}\nHistorial:  {hist_path}\nResumen:    {resumen_path}")


if __name__ == "__main__":
    main()
