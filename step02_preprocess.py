# -*- coding: utf-8 -*-
"""
Шаг 2. Предобработка стэков:
- загрузка out/stacks/<λ>nm/{stack.npy,z_um.npy}
- оценка индекса лучшего фокуса (резкость)
- выравнивание всех z-кадров к лучшему (FFT phase correlation, целочисленные сдвиги)
- (опц.) вычитание медианного фона по z
- нормировка
- сохранение в out/clean/<λ>nm/stack_clean.npy, idx_best.npy, shifts.csv
- QC графики: сдвиги vs z, превью лучшего кадра до/после

Зависимости: numpy, pandas, matplotlib
"""

import os
import math
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # неинтерактивный бэкенд
import matplotlib.pyplot as plt

# -------------------
# НАСТРОЙКИ
# -------------------
STACKS_DIR = Path("out") / "stacks"
CLEAN_DIR  = Path("out") / "clean"

# Вычитать медианный фон по z?
SUBTRACT_Z_MEDIAN = True

# Метрика резкости: "lapvar" или "grad"
SHARPNESS_METRIC = "lapvar"


# -------------------
# УТИЛИТЫ
# -------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def variance_of_laplacian(img: np.ndarray) -> float:
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    Fy = np.fft.rfft2(img, s=img.shape)
    Fk = np.fft.rfft2(k,   s=img.shape)
    conv = np.fft.irfft2(Fy * Fk, s=img.shape)
    return float(conv.var())

def gradient_energy(img: np.ndarray) -> float:
    gx = (np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)) / 2.0
    gy = (np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)) / 2.0
    g2 = gx * gx + gy * gy
    return float(g2.mean())

def sharpness(img: np.ndarray, metric: str = "lapvar") -> float:
    if metric == "grad":
        return gradient_energy(img)
    return variance_of_laplacian(img)

def phase_correlation_shift(ref: np.ndarray, mov: np.ndarray) -> Tuple[int, int]:
    """
    Возвращает целочисленные (dy, dx), куда надо сдвинуть mov, чтобы совместить с ref.
    """
    H, W = ref.shape
    F1 = np.fft.fft2(ref)
    F2 = np.fft.fft2(mov)
    R = F1 * np.conj(F2)
    denom = np.abs(R)
    denom[denom < 1e-9] = 1e-9
    R /= denom
    r = np.fft.ifft2(R)
    peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    p_y, p_x = int(peak[0]), int(peak[1])

    # Преобразуем координату пика в сдвиг с учётом торцевания
    dy = p_y if p_y <= H // 2 else p_y - H
    dx = p_x if p_x <= W // 2 else p_x - W
    return int(dy), int(dx)

def apply_shift_integer(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def plot_preview(img: np.ndarray, title: str, path: Path):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_shifts(z_um: np.ndarray, shifts: List[Tuple[int,int]], path: Path):
    dy = [s[0] for s in shifts]
    dx = [s[1] for s in shifts]
    plt.figure()
    plt.title("Сдвиги относительно лучшего фокуса")
    plt.plot(z_um, dy, label="dy (pix)")
    plt.plot(z_um, dx, label="dx (pix)")
    plt.xlabel("z (μm)")
    plt.ylabel("shift (pixels)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------
# ОСНОВА
# -------------------
def load_stacks(stacks_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Сканирует out/stacks/<λ>nm и грузит stack.npy, z_um.npy
    Возвращает словарь: {lambda_nm: {"stack": np.ndarray [Z,H,W], "z_um": np.ndarray}}
    """
    data: Dict[int, Dict[str, np.ndarray]] = {}
    if not stacks_dir.is_dir():
        print(f"⚠️ Нет каталога стэков: {stacks_dir}")
        return data
    for sub in sorted(stacks_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name.strip().lower()
        if not name.endswith("nm"):
            continue
        try:
            lam_nm = int(name[:-2])  # "595nm" -> 595
        except Exception:
            continue
        stack_path = sub / "stack.npy"
        z_path     = sub / "z_um.npy"
        if not (stack_path.exists() and z_path.exists()):
            continue
        stack = np.load(stack_path).astype(np.float32)  # [Z,H,W], уже нормировано в шаге 1
        z_um  = np.load(z_path).astype(np.float32)
        if stack.ndim != 3 or len(z_um) != stack.shape[0]:
            print(f"⚠️ Пропуск {sub}: форма не согласована")
            continue
        data[lam_nm] = {"stack": stack, "z_um": z_um}
    return data

def best_focus_index(stack: np.ndarray, metric: str) -> int:
    scores = [sharpness(stack[k], metric=metric) for k in range(stack.shape[0])]
    return int(np.argmax(scores))

def preprocess_lambda(lam_nm: int, stack: np.ndarray, z_um: np.ndarray, out_root: Path):
    """
    Выравниваем по сдвигам к лучшему фокусу, (опц.) вычитаем медиану по z, нормируем и сохраняем.
    """
    out_dir = out_root / f"{lam_nm}nm"
    ensure_dir(out_dir)

    # 1) индекс лучшего фокуса
    idx_best = best_focus_index(stack, metric=SHARPNESS_METRIC)
    ref = stack[idx_best]

    # 2) фазовая корреляция: сдвиги всех слоёв к ref
    shifts: List[Tuple[int,int]] = []
    aligned = np.empty_like(stack)
    for k in range(stack.shape[0]):
        mov = stack[k]
        dy, dx = (0, 0) if k == idx_best else phase_correlation_shift(ref, mov)
        shifts.append((dy, dx))
        aligned[k] = apply_shift_integer(mov, dy, dx)

    # 3) (опц.) вычитание медианного фона по z
    if SUBTRACT_Z_MEDIAN:
        median_bg = np.median(aligned, axis=0)
        aligned = aligned - median_bg[None, ...]
        aligned = np.clip(aligned, a_min=0.0, a_max=None)

    # 4) нормировка к максимуму по стэку
    mx = aligned.max()
    if mx > 0:
        aligned = aligned / mx

    # 5) сохранение
    np.save(out_dir / "stack_clean.npy", aligned.astype(np.float32))
    np.save(out_dir / "z_um.npy",      z_um.astype(np.float32))
    np.save(out_dir / "idx_best.npy",  np.array([idx_best], dtype=np.int32))

    # CSV сдвигов
    df_shifts = pd.DataFrame({"z_um": z_um, "dy_pix": [s[0] for s in shifts], "dx_pix": [s[1] for s in shifts]})
    df_shifts.to_csv(out_dir / "shifts.csv", index=False, encoding="utf-8-sig")

    # QC: график сдвигов и превью лучшего кадра «до/после»
    plot_shifts(z_um, shifts, out_dir / "shifts.png")
    plot_preview(stack[idx_best],   f"{lam_nm} nm — best (raw) z={z_um[idx_best]:.1f} μm", out_dir / "best_raw.png")
    plot_preview(aligned[idx_best], f"{lam_nm} nm — best (aligned)",                      out_dir / "best_aligned.png")

    print(f"λ={lam_nm} nm: clean stack → {out_dir} (best idx={idx_best}, z={z_um[idx_best]:.2f} μm)")

def main():
    ensure_dir(CLEAN_DIR)
    data = load_stacks(STACKS_DIR)
    if not data:
        print("⚠️ Стэков не найдено. Сначала выполните Шаг 1.")
        return
    for lam_nm in sorted(data.keys()):
        d = data[lam_nm]
        preprocess_lambda(lam_nm, d["stack"], d["z_um"], CLEAN_DIR)

if __name__ == "__main__":
    main()
