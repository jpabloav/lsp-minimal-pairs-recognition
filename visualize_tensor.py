import numpy as np
import pandas as pd

X = np.load("datos_procesados/stgcn_X_train.npy")
muestra = 0
persona = 0

m = X[muestra, :, :, :, persona]  # (3, 64, 75)

df = pd.DataFrame({
    "frame": np.repeat(np.arange(m.shape[1]), m.shape[2]),
    "joint": np.tile(np.arange(m.shape[2]), m.shape[1]),
    "x": m[0].ravel(),
    "y": m[1].ravel(),
    "z": m[2].ravel(),
})

df.to_csv("todos_los_joints_muestra_0.csv", index=False)
print("Archivo creado: todos_los_joints_muestra_0.csv")