"""
Utilidades compartidas por los scripts de entrenamiento, evaluación e
inferencia: registro de arquitecturas, convención de nombres de corridas y
parada temprana.

Las corridas se identifican como <arquitectura>_T<T>_s<semilla>
(por ejemplo, simplificado_T32_s42), y esa misma clave se usa para el
checkpoint, el historial de entrenamiento y la carpeta de resultados.
"""

import os

# Arquitecturas disponibles. Para agregar una nueva, se incluye su nombre
# aquí y su construcción en crear_modelo().
ARQUITECTURAS = ("simplificado",)


def crear_modelo(arq, num_classes, A):
    """Construye el modelo de la arquitectura indicada."""
    if arq == "simplificado":
        from stgcn_model import STGCN
        return STGCN(num_classes=num_classes, A=A)
    raise ValueError(f"Arquitectura desconocida: {arq!r}. Disponibles: {ARQUITECTURAS}")


def nombre_corrida(arq, T, semilla):
    """Clave de una corrida: <arq>_T<T>_s<semilla>."""
    return f"{arq}_T{T}_s{semilla}"


def ruta_checkpoint(arq, T, semilla, modelo_dir="modelo"):
    """Ruta del archivo de pesos de una corrida."""
    return os.path.join(modelo_dir, f"{nombre_corrida(arq, T, semilla)}.pt")


class ParadaTemprana:
    """
    Parada temprana sobre una métrica que se desea maximizar (F1-macro de
    validación). actualizar() devuelve (mejoro, detener): 'mejoro' indica que
    la métrica superó al mejor valor visto, y 'detener' que pasaron
    'paciencia' épocas consecutivas sin mejora.
    """

    def __init__(self, paciencia):
        self.paciencia = paciencia
        self.mejor = float("-inf")
        self.sin_mejora = 0

    def actualizar(self, valor):
        if valor > self.mejor:
            self.mejor = valor
            self.sin_mejora = 0
            return True, False
        self.sin_mejora += 1
        return False, self.sin_mejora >= self.paciencia
