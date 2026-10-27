"""
Preprocesamiento completo para ST-GCN con partición fija por sujetos y
longitud temporal T configurable.

Etapas:
    1. Indexa el dataset (build_dataset_index.py) y excluye las muestras
       listadas en la partición.
    2. Codifica las etiquetas de texto a números.
    3. Divide las muestras en train / val / test según los sujetos definidos
       en particion.json (ningún sujeto aparece en más de un conjunto).
    4. Preprocesa cada muestra a T frames (build_dataset_tensor de
       preprocess_stgcn.py).
    5. Guarda los tensores (X), las etiquetas (Y) y el sujeto de cada muestra
       (S) en <salida>/T<T>/ para no reprocesar en cada corrida de
       entrenamiento.

La partición vive en un archivo aparte para que otras arquitecturas puedan
usar exactamente la misma división de sujetos.

Uso:
    python run_split_and_preprocess.py --T 32
    python run_split_and_preprocess.py --T 16 --dataset Dataset --salida datos_procesados
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from build_dataset_index import build_file_index
from preprocess_stgcn import build_dataset_tensor, build_adjacency

OUTPUT_PREFIX = "stgcn"   # prefijo de los archivos de salida
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Preprocesamiento ST-GCN con partición fija por sujetos.")
    p.add_argument("--T", type=int, default=32, help="frames por muestra tras el remuestreo (default: 32)")
    p.add_argument("--dataset", default="Dataset", help="carpeta raíz del dataset (contiene las carpetas SUBJECT_n)")
    p.add_argument("--particion", default="particion.json", help="archivo JSON con la partición por sujetos")
    p.add_argument("--salida", default="datos_procesados", help="carpeta base de salida")
    return p.parse_args()


def load_partition(path):
    """Lee la partición y verifica que los conjuntos de sujetos sean disjuntos."""
    with open(path, "r", encoding="utf-8") as f:
        part = json.load(f)
    splits = {name: set(part[name]) for name in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = splits[a] & splits[b]
        if overlap:
            raise ValueError(f"Sujetos repetidos entre '{a}' y '{b}': {sorted(overlap)}")
    return splits, set(part.get("excluir_muestras", []))


def sample_key(path):
    """Identificador 'SUBJECT_n/Clase/sample_n' de una ruta .../landmarks/skeleton_*.txt."""
    parts = Path(path).parts
    return f"{parts[-5]}/{parts[-4]}/{parts[-3]}"


def main():
    args = parse_args()
    out_dir = os.path.join(args.salida, f"T{args.T}")
    os.makedirs(out_dir, exist_ok=True)
    splits, excluir = load_partition(args.particion)

    print("1) Indexando el dataset...")
    paths, labels, subjects = build_file_index(args.dataset, modality="real")
    print(f"   {len(paths)} muestras encontradas.")
    if not paths:
        raise SystemExit("No se encontraron muestras: revisar la ruta indicada en --dataset.")

    encontradas = {sample_key(p) for p in paths}
    faltantes = sorted(excluir - encontradas)
    if faltantes:
        print(f"   [aviso] muestras a excluir que no existen en el dataset: {faltantes}")
    keep = [i for i, p in enumerate(paths) if sample_key(p) not in excluir]
    print(f"   {len(paths) - len(keep)} muestra(s) excluida(s); se conservan {len(keep)}.")
    paths = [paths[i] for i in keep]
    labels = [labels[i] for i in keep]
    subjects = [subjects[i] for i in keep]

    asignados = set().union(*splits.values())
    sin_asignar = sorted(set(subjects) - asignados)
    if sin_asignar:
        print(f"   [aviso] sujetos del dataset sin conjunto asignado (se ignoran): {sin_asignar}")
    inexistentes = sorted(asignados - set(subjects))
    if inexistentes:
        print(f"   [aviso] sujetos de la partición que no existen en el dataset: {inexistentes}")

    print("2) Codificando etiquetas...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    np.save(f"{out_dir}/{OUTPUT_PREFIX}_label_classes.npy", encoder.classes_)
    print(f"   {len(encoder.classes_)} clases codificadas.")

    print("3) Dividiendo por sujeto según la partición...")
    subjects_arr = np.array(subjects)
    idx = {name: np.where(np.isin(subjects_arr, sorted(splits[name])))[0] for name in SPLITS}
    for name in SPLITS:
        ii = idx[name]
        if len(ii) == 0:
            raise SystemExit(f"El conjunto '{name}' quedó vacío: revisar la partición.")
        print(f"   {name:5s}: {len(ii)} muestras, {len(set(y[ii]))} clases, "
              f"sujetos {sorted(set(map(str, subjects_arr[ii])))}")

    print(f"4) Preprocesando a T={args.T} (imputación + normalización + remuestreo)...")
    for name in SPLITS:
        ii = idx[name]
        X, _ = build_dataset_tensor([paths[i] for i in ii], T_target=args.T)
        np.save(f"{out_dir}/{OUTPUT_PREFIX}_X_{name}.npy", X)
        np.save(f"{out_dir}/{OUTPUT_PREFIX}_Y_{name}.npy", y[ii])
        np.save(f"{out_dir}/{OUTPUT_PREFIX}_S_{name}.npy", subjects_arr[ii])  # sujeto de cada muestra
        print(f"   X_{name}: {X.shape}  Y_{name}: {y[ii].shape}")

    print("5) Guardando matriz de adyacencia y metadatos...")
    np.save(f"{out_dir}/{OUTPUT_PREFIX}_adjacency.npy", build_adjacency())
    meta = {
        "T": args.T,
        "muestras": {name: int(len(idx[name])) for name in SPLITS},
        "sujetos": {name: sorted(splits[name]) for name in SPLITS},
        "excluidas": sorted(excluir),
    }
    with open(f"{out_dir}/{OUTPUT_PREFIX}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Archivos generados en {out_dir}/")


if __name__ == "__main__":
    main()
