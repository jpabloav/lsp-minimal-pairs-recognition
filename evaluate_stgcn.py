"""
Evaluación de un modelo ST-GCN entrenado sobre el conjunto de prueba (o de
validación) de la partición por sujetos.

Entradas:
    <datos>/T<T>/stgcn_X_<conjunto>.npy, stgcn_Y_<conjunto>.npy,
    stgcn_adjacency.npy, stgcn_label_classes.npy
    <modelo-dir>/<arq>_T<T>_s<semilla>.pt   (generado por train_stgcn.py)

Salidas, en <resultados>/<arq>_T<T>_s<semilla>/<conjunto>/:
    - metricas_por_clase.csv      precision, recall, F1 y soporte por clase
    - matriz_confusion.png        matriz de confusión normalizada por fila
    - matriz_confusion.csv        matriz de confusión con conteos (filas: clase
                                  real; columnas: clase predicha)
    - confusiones.csv             pares (clase real -> clase predicha) con sus
                                  conteos, ordenados de mayor a menor
    - resumen.json                métricas globales, desempeño por sujeto y
                                  análisis por grupos de minimal pairs (útil
                                  para comparar corridas)
Además imprime las métricas por consola. El desempeño por sujeto requiere el
archivo stgcn_S_<conjunto>.npy generado por run_split_and_preprocess.py.

Uso:
    python evaluate_stgcn.py --T 32
    python evaluate_stgcn.py --T 32 --conjunto val
"""

import argparse
import json
import os

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

from modelos import ARQUITECTURAS, nombre_corrida, ruta_checkpoint

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluación ST-GCN para LSP.")
    p.add_argument("--T", type=int, default=32, help="frames por muestra de la corrida a evaluar")
    p.add_argument("--arq", default="simplificado", choices=ARQUITECTURAS, help="arquitectura")
    p.add_argument("--semilla", type=int, default=42, help="semilla de la corrida a evaluar")
    p.add_argument("--conjunto", default="test", choices=("test", "val"),
                   help="conjunto a evaluar (test para el resultado final; val solo diagnóstico)")
    p.add_argument("--datos", default="datos_procesados", help="carpeta base de los datos preprocesados")
    p.add_argument("--modelo-dir", default="modelo", help="carpeta de los modelos entrenados")
    p.add_argument("--resultados", default="resultados", help="carpeta base de salida")
    return p.parse_args(argv)


def load_model_and_data(args):
    import torch
    from modelos import crear_modelo

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = os.path.join(args.datos, f"T{args.T}")
    X = np.load(os.path.join(d, f"stgcn_X_{args.conjunto}.npy"))
    y = np.load(os.path.join(d, f"stgcn_Y_{args.conjunto}.npy"))
    A = np.load(os.path.join(d, "stgcn_adjacency.npy"))
    classes = np.load(os.path.join(d, "stgcn_label_classes.npy"), allow_pickle=True)
    ruta_s = os.path.join(d, f"stgcn_S_{args.conjunto}.npy")
    subjects = np.load(ruta_s) if os.path.exists(ruta_s) else None

    ckpt = ruta_checkpoint(args.arq, args.T, args.semilla, args.modelo_dir)
    if not os.path.exists(ckpt):
        raise SystemExit(f"No existe el checkpoint {ckpt}. Entrenar con: "
                         f"python train_stgcn.py --T {args.T} --semilla {args.semilla}")
    model = crear_modelo(args.arq, len(classes), A).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device, X, y, classes, subjects


def predict_all(model, device, X, batch_size=BATCH_SIZE):
    import torch

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def global_metrics(y_true, y_pred, conjunto="test"):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"=== Métricas globales ({conjunto}) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-macro:    {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    return acc, f1_macro, f1_weighted


def per_class_report(y_true, y_pred, classes, csv_path, conjunto="test"):
    labels = np.arange(len(classes))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(classes),
        zero_division=0, output_dict=True,
    )
    print(f"\n=== Reporte por clase ({conjunto}) ===")
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


def plot_confusion_matrix(y_true, y_pred, classes, out_path, conjunto="test"):
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
    ax.set_title(f"Matriz de confusión normalizada por fila ({conjunto})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusión guardada en {out_path}")


def save_confusion_csv(y_true, y_pred, classes, out_path):
    """Matriz de confusión con conteos: filas = clase real, columnas = clase predicha."""
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("real\\predicha," + ",".join(classes) + "\n")
        for i, c in enumerate(classes):
            f.write(c + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")
    print(f"Matriz de confusión (conteos) guardada en {out_path}")
    return cm


def top_confusions(cm, classes, out_path, recall_umbral=0.5, k=3):
    """
    Guarda todos los pares (clase real -> clase predicha) con errores, ordenados
    por cantidad, e imprime hacia qué clases se van los errores de las clases
    con recall menor a recall_umbral.
    """
    n = len(classes)
    soporte = cm.sum(axis=1)
    pares = [(cm[i, j], i, j) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pares.sort(reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("clase_real,clase_predicha,cantidad,porcentaje_de_la_clase_real\n")
        for cant, i, j in pares:
            f.write(f"{classes[i]},{classes[j]},{int(cant)},{cant / soporte[i]:.3f}\n")
    print(f"Confusiones guardadas en {out_path}")

    print(f"\n=== Destino de los errores en clases con recall < {recall_umbral} ===")
    for i in range(n):
        if soporte[i] == 0:
            continue
        recall = cm[i, i] / soporte[i]
        if recall >= recall_umbral:
            continue
        destinos = sorted(((cm[i, j], j) for j in range(n) if j != i and cm[i, j] > 0), reverse=True)[:k]
        texto = ", ".join(f"{classes[j]}={int(c)}" for c, j in destinos) or "(sin errores)"
        print(f"  {classes[i]} (n={int(soporte[i])}, recall={recall:.2f}, correctas={int(cm[i, i])}) -> {texto}")


def per_subject_metrics(y_true, y_pred, subjects, conjunto="test"):
    """Accuracy y F1-macro calculados por separado para cada sujeto."""
    resultados = {}
    print(f"\n=== Desempeño por sujeto ({conjunto}) ===")
    for s in sorted(set(map(str, subjects)), key=lambda x: int(x.split("_")[-1])):
        m = np.asarray(subjects).astype(str) == s
        acc = accuracy_score(y_true[m], y_pred[m])
        f1 = f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)
        resultados[s] = {"muestras": int(m.sum()), "accuracy": float(acc), "f1_macro": float(f1)}
        print(f"  {s:11s} muestras={int(m.sum()):4d}  accuracy={acc:.4f}  F1-macro={f1:.4f}")
    return resultados


def minimal_pair_analysis(y_true, y_pred, classes, groups=MINIMAL_PAIR_GROUPS, conjunto="test"):
    """
    Para cada grupo de minimal pairs reporta, sobre las muestras cuya clase
    real pertenece al grupo: acierto dentro del grupo, y cuántos errores
    fueron confusiones con otra clase del mismo grupo frente a errores hacia
    clases externas. Devuelve la misma información como lista de diccionarios.
    """
    name_to_idx = {c: i for i, c in enumerate(classes)}
    resultados = []
    print(f"\n=== Análisis por grupos de minimal pairs ({conjunto}) ===")
    for group in groups:
        missing = [g for g in group if g not in name_to_idx]
        if missing:
            print(f"\nGrupo {group}: omitido, clases no encontradas: {missing}")
            continue

        idxs = [name_to_idx[g] for g in group]
        mask = np.isin(y_true, idxs)
        n = int(mask.sum())
        if n == 0:
            print(f"\nGrupo {group}: sin muestras en {conjunto}")
            continue

        yt, yp = y_true[mask], y_pred[mask]
        correct = int((yt == yp).sum())
        within = int(((yt != yp) & np.isin(yp, idxs)).sum())
        outside = int(((yt != yp) & ~np.isin(yp, idxs)).sum())

        print(f"\nGrupo {group}  ({n} muestras)")
        print(f"  Acierto dentro del grupo:      {correct}/{n} ({correct / n:.1%})")
        print(f"  Errores hacia otra clase del grupo: {within}")
        print(f"  Errores hacia clases externas:      {outside}")
        detalle = {}
        for a in group:
            ia = name_to_idx[a]
            sub = yt == ia
            if sub.sum() == 0:
                continue
            fila = {classes[j]: int((yp[sub] == j).sum()) for j in idxs}
            detalle[a] = {"muestras": int(sub.sum()), "predicho_en_grupo": fila}
            print(f"    real {a} ({int(sub.sum())}) -> "
                  + ", ".join(f"{k}={v}" for k, v in fila.items()))

        resultados.append({
            "grupo": list(group), "muestras": n, "aciertos": correct,
            "errores_dentro_del_grupo": within, "errores_hacia_externas": outside,
            "detalle": detalle,
        })
    return resultados


def main(argv=None):
    args = parse_args(argv)
    nombre = nombre_corrida(args.arq, args.T, args.semilla)
    out_dir = os.path.join(args.resultados, nombre, args.conjunto)
    os.makedirs(out_dir, exist_ok=True)

    model, device, X, y, classes, subjects = load_model_and_data(args)
    print(f"Corrida: {nombre}  |  Conjunto: {args.conjunto}  |  Muestras: {len(y)}  |  "
          f"Clases: {len(classes)}  |  Dispositivo: {device}\n")

    y_pred = predict_all(model, device, X)

    acc, f1_macro, f1_weighted = global_metrics(y, y_pred, args.conjunto)
    per_class_report(y, y_pred, classes, os.path.join(out_dir, "metricas_por_clase.csv"), args.conjunto)
    plot_confusion_matrix(y, y_pred, classes, os.path.join(out_dir, "matriz_confusion.png"), args.conjunto)
    cm = save_confusion_csv(y, y_pred, classes, os.path.join(out_dir, "matriz_confusion.csv"))
    top_confusions(cm, classes, os.path.join(out_dir, "confusiones.csv"))
    if subjects is not None:
        por_sujeto = per_subject_metrics(y, y_pred, subjects, args.conjunto)
    else:
        por_sujeto = None
        print("\n[aviso] No se encontró stgcn_S_%s.npy: se omite el desempeño por sujeto. "
              "Regenerar los datos con run_split_and_preprocess.py." % args.conjunto)
    grupos = minimal_pair_analysis(y, y_pred, classes, conjunto=args.conjunto)

    resumen = {
        "corrida": nombre, "arquitectura": args.arq, "T": args.T, "semilla": args.semilla,
        "conjunto": args.conjunto, "muestras": int(len(y)),
        "accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted),
        "por_sujeto": por_sujeto,
        "grupos_minimal_pairs": grupos,
    }
    with open(os.path.join(out_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {out_dir}/")


if __name__ == "__main__":
    main()
