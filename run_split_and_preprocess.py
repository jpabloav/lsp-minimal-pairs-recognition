"""
Ejecuta el pipeline completo de preprocesamiento ST-GCN, de punta a punta:

    1. Indexa el dataset (build_dataset_index.py)
    2. Codifica las etiquetas de texto a números
    3. Separa train/test agrupando por sujeto (nadie se repite entre splits)
    4. Preprocesa cada muestra (build_dataset_tensor de preprocess_stgcn.py)
    5. Guarda los tensores resultantes en disco (.npy), evitando reprocesar
       en cada corrida de entrenamiento

Requiere que build_dataset_index.py y preprocess_stgcn.py estén en la
misma carpeta que este script, y las librerías scikit-learn y numpy
instaladas (`pip install scikit-learn numpy`).

Uso:
    python run_split_and_preprocess.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from build_dataset_index import build_file_index
from preprocess_stgcn import build_dataset_tensor, build_adjacency

# --- Configuración -------------------------------------------------------
DATASET_ROOT = r"Dataset\Dataset"   # carpeta raíz del dataset descomprimido
T_TARGET = 64               # frames fijos por muestra tras el remuestreo
TEST_SIZE = 0.2             # proporción de SUJETOS (no muestras) para test
RANDOM_STATE = 42
OUTPUT_PREFIX = "stgcn"     # prefijo de los .npy de salida
# ------------------------------------------------------------------------


def main():
    print("1) Indexando el dataset...")
    paths, labels, subjects = build_file_index(DATASET_ROOT, modality="real")
    print(f"   {len(paths)} muestras encontradas.")

    print("2) Codificando etiquetas...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    np.save(f"{OUTPUT_PREFIX}_label_classes.npy", encoder.classes_)
    print(f"   {len(encoder.classes_)} clases codificadas "
          f"(guardadas en {OUTPUT_PREFIX}_label_classes.npy).")

    print("3) Separando train/test por sujeto...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(paths, y, groups=subjects))

    paths_train = [paths[i] for i in train_idx]
    paths_test = [paths[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    subjects_train = sorted(set(subjects[i] for i in train_idx))
    subjects_test = sorted(set(subjects[i] for i in test_idx))
    print(f"   Train: {len(paths_train)} muestras, sujetos {subjects_train}")
    print(f"   Test:  {len(paths_test)} muestras, sujetos {subjects_test}")

    print("4) Preprocesando train (interpolación + normalización + remuestreo)...")
    X_train, _ = build_dataset_tensor(paths_train, T_target=T_TARGET)
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")

    print("5) Preprocesando test...")
    X_test, _ = build_dataset_tensor(paths_test, T_target=T_TARGET)
    print(f"   X_test: {X_test.shape}  y_test: {y_test.shape}")

    print("6) Construyendo matriz de adyacencia del grafo...")
    A = build_adjacency()

    print("7) Guardando resultados en disco...")
    np.save(f"{OUTPUT_PREFIX}_X_train.npy", X_train)
    np.save(f"{OUTPUT_PREFIX}_y_train.npy", y_train)
    np.save(f"{OUTPUT_PREFIX}_X_test.npy", X_test)
    np.save(f"{OUTPUT_PREFIX}_y_test.npy", y_test)
    np.save(f"{OUTPUT_PREFIX}_adjacency.npy", A)

    print("\nListo. Archivos generados:")
    for suf in ["X_train", "y_train", "X_test", "y_test", "adjacency", "label_classes"]:
        print(f"  {OUTPUT_PREFIX}_{suf}.npy")


if __name__ == "__main__":
    main()
