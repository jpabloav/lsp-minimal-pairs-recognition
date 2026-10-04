"""
Preprocesamiento de landmarks para ST-GCN (reconocimiento de LSP)
===================================================================

Formato de entrada: archivos skeleton_real.txt con 75 landmarks por
frame (33 de pose de MediaPipe + 21 de mano izquierda + 21 de mano
derecha), 3 coordenadas por landmark (x, y, z) normalizadas por
MediaPipe. Cada línea del archivo corresponde a un frame; cada frame
contiene 225 valores (75 * 3) separados por espacios.

Pipeline:
    1. Carga del archivo -> array (T, 75, 3)
    2. Imputación de landmarks faltantes (NaN) en el eje temporal
    3. Normalización espacial (centrado en hombros + escala por
       distancia entre hombros), para hacer al esqueleto invariante a
       la posición del signante frente a la cámara
    4. Remuestreo temporal a un T fijo (debe coincidir con el T usado
       en la rama CNN-LSTM, para mantener comparabilidad entre
       arquitecturas)
    5. Construcción del tensor final (N, C, T, V, M) esperado por ST-GCN
    6. Matriz de adyacencia del grafo (pose superior + 2 manos)

Índices de referencia de MediaPipe Pose:
    11 = hombro izquierdo   12 = hombro derecho
    13 = codo izquierdo     14 = codo derecho
    15 = muñeca izquierda   16 = muñeca derecha
    23 = cadera izquierda   24 = cadera derecha
"""

import numpy as np

# ---------------------------------------------------------------------------
# Configuración del grafo
# ---------------------------------------------------------------------------
N_POSE = 33
N_HAND = 21
N_JOINTS = N_POSE + 2 * N_HAND  # 75

LEFT_HAND_START = N_POSE              # 33
RIGHT_HAND_START = N_POSE + N_HAND    # 54

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


def _pose_edges():
    """Conexiones del tren superior relevantes para lengua de señas."""
    return [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
    ]


def _hand_edges(offset):
    """Conexiones estándar de un esqueleto de mano de 21 puntos (índice 0 = muñeca)."""
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
        (0, 5), (5, 6), (6, 7), (7, 8),          # índice
        (0, 9), (9, 10), (10, 11), (11, 12),     # medio
        (0, 13), (13, 14), (14, 15), (15, 16),   # anular
        (0, 17), (17, 18), (18, 19), (19, 20),   # meñique
        (5, 9), (9, 13), (13, 17),               # palma
    ]
    return [(a + offset, b + offset) for a, b in base_edges]


def build_adjacency():
    """Matriz de adyacencia (75x75) del grafo pose-superior + mano izq + mano der."""
    edges = _pose_edges()
    edges += _hand_edges(LEFT_HAND_START)
    edges += _hand_edges(RIGHT_HAND_START)
    # conecta la muñeca de la pose con la muñeca del esqueleto de mano detallado
    edges += [(LEFT_WRIST, LEFT_HAND_START), (RIGHT_WRIST, RIGHT_HAND_START)]

    A = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    np.fill_diagonal(A, 1)  # self-loops, estándar en ST-GCN
    return A


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_skeleton_real(path, n_joints=N_JOINTS, n_coords=3):
    """Carga un archivo skeleton_real.txt -> array (T, n_joints, n_coords)."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    T = data.shape[0]
    return data.reshape(T, n_joints, n_coords)


# ---------------------------------------------------------------------------
# Limpieza: interpolación de NaN (manos no detectadas en algunos frames)
# ---------------------------------------------------------------------------
def missing_ratio(joints):
    """Porcentaje de valores NaN en la muestra original, antes de imputar."""
    return float(np.isnan(joints).mean())


def interpolate_missing(joints, max_gap_frames=5):
    """
    Imputa NaN combinando dos estrategias según la longitud del hueco
    (huecos = frames consecutivos donde ese joint no se detectó):

      - Huecos cortos (<= max_gap_frames, con un frame válido antes Y
        después): interpolación lineal temporal entre el último valor
        válido antes del hueco y el primero después. Preserva la
        continuidad del movimiento sin saltos abruptos.

      - Huecos largos (con al menos un lado válido): forward-fill
        (o backward-fill si el hueco está al inicio del clip y no hay
        "antes") -- se repite de forma CONSTANTE el último frame válido,
        no un ciclo de varios frames, para no inventar un movimiento
        periódico que nunca ocurrió.

      - Un joint ausente en TODO el clip (ningún frame válido para
        anclar el forward/backward-fill): se rellena con 0, único caso
        sin alternativa razonable.
    """
    T, V, C = joints.shape
    joints = joints.copy()

    for v in range(V):
        for c in range(C):
            series = joints[:, v, c]
            nan_mask = np.isnan(series)
            if not nan_mask.any():
                continue

            if nan_mask.all():
                # el joint nunca se detectó en todo el clip -> cero
                series[:] = 0.0
                joints[:, v, c] = series
                continue

            # localizar los huecos (tramos consecutivos de NaN)
            padded = np.r_[False, nan_mask, False]
            changes = np.diff(padded.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for s, e in zip(starts, ends):
                gap_len = e - s
                has_before = s > 0
                has_after = e < T
                if gap_len <= max_gap_frames and has_before and has_after:
                    # hueco corto con ambos lados válidos -> interpolación lineal
                    t1, t2 = s - 1, e
                    q1, q2 = series[t1], series[t2]
                    frac = (np.arange(s, e) - t1) / (t2 - t1)
                    series[s:e] = q1 + frac * (q2 - q1)
                elif has_before:
                    # hueco largo, o pegado al final del clip -> forward-fill:
                    # repetir el último frame válido de forma constante
                    series[s:e] = series[s - 1]
                else:
                    # hueco pegado al inicio del clip (sin "antes") ->
                    # backward-fill: repetir hacia atrás el primer frame válido
                    series[s:e] = series[e]

            joints[:, v, c] = series

    return joints


# ---------------------------------------------------------------------------
# Normalización espacial
# ---------------------------------------------------------------------------
def normalize_spatial(joints, ref_a=LEFT_SHOULDER, ref_b=RIGHT_SHOULDER):
    """
    Centra cada frame en el punto medio entre ref_a y ref_b (hombros por
    defecto) y escala por la distancia entre esos dos puntos. Esto hace al
    esqueleto invariante a la posición del signante frente a la cámara y a
    diferencias de tamaño/distancia entre distintos sujetos.
    """
    center_x = (joints[:, ref_a, 0] + joints[:, ref_b, 0]) / 2  # (T,)
    center_y = (joints[:, ref_a, 1] + joints[:, ref_b, 1]) / 2  # (T,)
    scale = np.linalg.norm(
        joints[:, ref_a, :2] - joints[:, ref_b, :2], axis=1
    )  # (T,)
    scale = np.where(scale < 1e-6, 1.0, scale)  # evita división por cero

    norm = joints.copy()
    norm[:, :, 0] = (joints[:, :, 0] - center_x[:, None]) / scale[:, None]
    norm[:, :, 1] = (joints[:, :, 1] - center_y[:, None]) / scale[:, None]
    if joints.shape[2] == 3:
        # z ya viene relativa (MediaPipe); solo se reescala con el mismo factor
        norm[:, :, 2] = joints[:, :, 2] / scale[:, None]
    return norm


# ---------------------------------------------------------------------------
# Remuestreo temporal a T fijo
# ---------------------------------------------------------------------------
def resample_temporal(joints, T_target):
    """Remuestrea la secuencia a T_target frames por interpolación lineal en el tiempo."""
    T, V, C = joints.shape
    if T == T_target:
        return joints
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, T_target)
    out = np.empty((T_target, V, C), dtype=joints.dtype)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(new_t, old_t, joints[:, v, c])
    return out


# ---------------------------------------------------------------------------
# Pipeline completo por muestra y armado del tensor del dataset
# ---------------------------------------------------------------------------
def process_sample(path, T_target=64):
    """
    Pipeline completo para una muestra:
    carga -> interpola NaN -> normaliza espacialmente -> remuestrea a T_target.
    Devuelve (array (C, T_target, V), ratio_de_nan_original).
    """
    joints = load_skeleton_real(path)
    ratio = missing_ratio(joints)
    joints = interpolate_missing(joints)
    joints = normalize_spatial(joints)
    joints = resample_temporal(joints, T_target)
    joints = np.transpose(joints, (2, 0, 1))  # (T,V,C) -> (C,T,V)
    return joints, ratio


def build_dataset_tensor(paths, T_target=64):
    """
    Procesa una lista de rutas a archivos skeleton_real.txt y arma el tensor
    final (N, C, T, V, M) esperado por ST-GCN. M=1 porque cada clip tiene un
    solo firmante.

    No se descarta ninguna muestra por porcentaje de NaN: la literatura de
    reconocimiento de señas basado en esqueletos generalmente imputa los
    valores faltantes en vez de descartar la muestra completa (ver
    interpolate_missing). Aquí solo se imputa.

    Devuelve (X, missing_ratios): missing_ratios contiene el porcentaje de
    NaN original de cada muestra (antes de imputar), útil como estadística
    de calidad de datos.
    """
    samples = []
    missing_ratios = np.zeros(len(paths), dtype=float)
    for i, p in enumerate(paths):
        joints, ratio = process_sample(p, T_target)
        missing_ratios[i] = ratio
        samples.append(joints)

    X = np.stack(samples, axis=0)      # (N, C, T, V)
    X = X[:, :, :, :, None]            # (N, C, T, V, M=1)

    high_missing = np.where(missing_ratios > 0.3)[0]
    if len(high_missing):
        print(f"[info] {len(high_missing)} muestra(s) con >30% de NaN "
              f"(se conservan; reportadas para control de calidad de datos):")
        for i in high_missing:
            print(f"  - {paths[i]} ({missing_ratios[i]:.1%} de valores NaN)")

    return X, missing_ratios


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ejemplo = "ruta/a/una/muestra/skeleton_real.txt"

    joints_raw = load_skeleton_real(ejemplo)
    print("Forma original (T, V, C):", joints_raw.shape)
    print(f"% de valores NaN antes de imputar: {missing_ratio(joints_raw):.2%}")

    procesado, ratio = process_sample(ejemplo, T_target=64)
    print("Forma final por muestra (C, T, V):", procesado.shape)
    print(f"% de NaN original de esta muestra: {ratio:.2%}")

    A = build_adjacency()
    print("Matriz de adyacencia:", A.shape, "- conexiones totales:", int(A.sum() - N_JOINTS))

    # Ejemplo de armado de tensor con una sola muestra repetida
    X, _ = build_dataset_tensor([ejemplo, ejemplo], T_target=64)
    print("Tensor final del dataset (N, C, T, V, M):", X.shape)
