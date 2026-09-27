"""
Indexado del dataset de LSP
============================

Recorre la estructura:

    Dataset/SUBJECT_n/Nombre_de_la_palabra/sample_n/landmarks/skeleton_real.txt

y arma tres listas paralelas (misma posición = misma muestra):
    - paths:    ruta al archivo skeleton_real.txt (o skeleton_pixel.txt)
    - labels:   nombre de la clase (carpeta "Nombre_de_la_palabra")
    - subjects: identificador del sujeto (carpeta "SUBJECT_n")

Esto es lo que luego se le pasa a build_dataset_tensor() del script
preprocess_stgcn.py, y a la codificación de labels / split por sujeto.
"""

from pathlib import Path
from collections import Counter


def build_file_index(dataset_root, modality="real"):
    """
    dataset_root: ruta a la carpeta Dataset/ (la que contiene los SUBJECT_n)
    modality: "real" (skeleton_real.txt) o "pixel" (skeleton_pixel.txt)
    """
    dataset_root = Path(dataset_root)
    filename = f"skeleton_{modality}.txt"

    paths, labels, subjects = [], [], []

    # patrón: SUBJECT_n / Clase / sample_n / landmarks / skeleton_*.txt
    for path in sorted(dataset_root.glob(f"*/*/*/landmarks/{filename}")):
        sample_dir = path.parent.parent     # sample_n
        class_dir = sample_dir.parent       # Nombre_de_la_palabra
        subject_dir = class_dir.parent      # SUBJECT_n

        paths.append(str(path))
        labels.append(class_dir.name)
        subjects.append(subject_dir.name)

    return paths, labels, subjects


def summarize(paths, labels, subjects):
    print(f"Total de muestras encontradas: {len(paths)}")
    print(f"Clases únicas: {len(set(labels))}")
    print(f"Sujetos únicos: {len(set(subjects))}")

    counts = Counter(labels)
    print("\nMuestras por clase (ordenado de menor a mayor):")
    for clase, n in sorted(counts.items(), key=lambda x: x[1]):
        print(f"  {clase:30s} {n}")


if __name__ == "__main__":
    # Ruta raíz del dataset descomprimido (contiene las carpetas SUBJECT_n)
    DATASET_ROOT = r"Dataset\Dataset"

    paths, labels, subjects = build_file_index(DATASET_ROOT, modality="real")
    summarize(paths, labels, subjects)

    if paths:
        print("\nEjemplo de una muestra indexada:")
        print("  path:   ", paths[0])
        print("  label:  ", labels[0])
        print("  subject:", subjects[0])
