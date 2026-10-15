
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_stgcn import load_skeleton_real, process_sample

# --- Carpeta de salida ---
OUT = "resultados_analisis"
os.makedirs(OUT, exist_ok=True)

D = "datos_procesados"
X = np.load(f"{D}/stgcn_X_train.npy")
y = np.load(f"{D}/stgcn_Y_train.npy")
classes = np.load(f"{D}/stgcn_label_classes.npy", allow_pickle=True)

print("Forma X_train:", X.shape, "| tipo:", X.dtype)
print("NaN restantes:", int(np.isnan(X).sum()))
print("Rango x,y,z (min/max):",
      [(float(X[:, c].min()), float(X[:, c].max())) for c in range(3)])

# --- CSV ---
i = 0
m = X[i, :, :, :, 0]  # (3, 64, 75)

df = pd.DataFrame({
    "frame": np.repeat(np.arange(64), 75),
    "joint": np.tile(np.arange(75), 64),
    "x": m[0].ravel(),
    "y": m[1].ravel(),
    "z": m[2].ravel(),
})

csv_path = os.path.join(OUT, "muestra_preprocesada.csv")
df.to_csv(csv_path, index=False)

print("Clase de la muestra:", classes[y[i]])
print("CSV guardado en:", csv_path)

# --- Antes vs. después ---
RAW_PATH = r"Dataset/SUBJECT_1/Poor/sample_1/landmarks/skeleton_real.txt"

raw = load_skeleton_real(RAW_PATH)
proc, _ = process_sample(RAW_PATH, 64)

print("NaN en crudo:", int(np.isnan(raw).sum()),
      "| frames originales:", raw.shape[0])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

ax[0].plot(raw[:, 16, 0])
ax[0].set_title(f"Antes (T={raw.shape[0]}, con NaN)")

ax[1].plot(proc[0, :, 16])
ax[1].set_title("Después (T=64, normalizado)")

for a in ax:
    a.set_xlabel("Frame")

plt.tight_layout()

img_path = os.path.join(OUT, "antes_despues.png")
plt.savefig(img_path, dpi=150, bbox_inches="tight")
plt.close()

print("Imagen guardada en:", img_path)
print("Análisis completado.")
