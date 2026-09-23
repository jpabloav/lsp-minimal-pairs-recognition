"""
Arquitectura ST-GCN (Spatial Temporal Graph Convolutional Network)
para clasificar las 34 clases de LSP.

Basado en Yan et al. 2018 "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition", adaptado al grafo de 75 joints
(pose superior + 2 manos) de preprocess_stgcn.py.

Entrada esperada: tensor (N, C, T, V, M) generado por
run_split_and_preprocess.py:
    N = número de muestras
    C = 3 (x, y, z)
    T = 64 (frames fijos)
    V = 75 (joints)
    M = 1 (un solo firmante por clip)
"""

import numpy as np
import torch
import torch.nn as nn


def normalize_adjacency(A):
    """
    Normalización simétrica D^-1/2 A D^-1/2 (estándar en GCN). A ya debe
    traer self-loops incluidos (build_adjacency() de preprocess_stgcn.py
    ya los agrega con np.fill_diagonal), así que aquí NO se le vuelve a
    sumar la identidad.
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.zeros_like(D)
    D_inv_sqrt[D > 0] = np.power(D[D > 0], -0.5)
    D_mat = np.diag(D_inv_sqrt)
    return D_mat @ A @ D_mat


class SpatialGraphConv(nn.Module):
    """Agrega información entre joints conectados (según A) dentro de cada frame."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        x = self.conv(x)                                # (N, C_out, T, V)
        x = torch.einsum("nctv,vw->nctw", x, self.A)     # propagación sobre el grafo
        return x


class STGCNBlock(nn.Module):
    """Bloque ST-GCN: conv espacial (grafo) + conv temporal + conexión residual."""

    def __init__(self, in_channels, out_channels, A, temporal_kernel=9, stride=1, residual=True):
        super().__init__()
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)

        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x) if self.residual is not None else 0
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.bn2(self.temporal_conv(x))
        return self.relu(x + res)


class STGCN(nn.Module):
    """Red ST-GCN completa: pila de bloques + pooling global + clasificador lineal."""

    def __init__(self, num_classes, A, in_channels=3, base_channels=64):
        super().__init__()
        A_norm = normalize_adjacency(A)
        V = A.shape[0]
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        c = base_channels
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, c, A_norm, stride=1, residual=False),
            STGCNBlock(c, c, A_norm, stride=1),
            STGCNBlock(c, c, A_norm, stride=1),
            STGCNBlock(c, c * 2, A_norm, stride=2),      # reduce T a la mitad
            STGCNBlock(c * 2, c * 2, A_norm, stride=1),
            STGCNBlock(c * 2, c * 4, A_norm, stride=2),  # reduce T de nuevo
            STGCNBlock(c * 4, c * 4, A_norm, stride=1),
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        # x: (N, C, T, V, M) -> nos quedamos con M=1
        N, C, T, V, M = x.shape
        x = x[:, :, :, :, 0]                                          # (N, C, T, V)

        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)      # (N, C*V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()       # (N, C, T, V)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x)                                              # (N, C_final, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    # Prueba rápida con datos sintéticos para verificar que las formas cuadran
    A = np.load("stgcn_adjacency.npy")
    model = STGCN(num_classes=34, A=A)
    x_dummy = torch.randn(4, 3, 64, 75, 1)   # batch de 4 muestras
    out = model(x_dummy)
    print("Entrada:", x_dummy.shape)
    print("Salida (logits por clase):", out.shape)  # esperado: (4, 34)