# -*- coding: utf-8 -*-
"""
Шаг 4A. Свод фокусов Z(λ) и кривых резкости из out/clean/<λ>nm.
Считает несколько метрик резкости, строит графики и summary.csv.

Вход:  out/clean/<λ>nm/{stack_clean.npy, z_um.npy}
Выход: out/focus/{λ}nm_sharpness.png, summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_DIR = Path("out") / "clean"
OUT_DIR   = Path("out") / "focus"

# --- Метрики резкости ---
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
    g2 = gx*gx + gy*gy
    return float(g2.mean())

def highfreq_fft_energy(img: np.ndarray, frac: float = 0.25) -> float:
    # энергия в высоких частотах (внешнее кольцо спектра)
    F = np.fft.fftshift(np.fft.fft2(img))
    mag2 = (np.abs(F)**2)
    H, W = img.shape
    y, x = np.ogrid[:H,:W]
    cy, cx = H//2, W//2
    r = np.sqrt((y-cy)**2 + (x-cx)**2)
    rmax = r.max()
    mask = (r > frac*rmax)
    return float(mag2[mask].mean())

def sharpness_curve(stack: np.ndarray, kind: str) -> np.ndarray:
    vals = []
    for k in range(stack.shape[0]):
        img = stack[k]
        if kind == "lapvar":
            vals.append(variance_of_laplacian(img))
        elif kind == "grad":
            vals.append(gradient_energy(img))
        else:
            vals.append(highfreq_fft_energy(img))
    return np.asarray(vals, dtype=np.float64)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dir(OUT_DIR)
    rows = []
    if not CLEAN_DIR.is_dir():
        print(f"⚠️ Нет каталога {CLEAN_DIR}")
        return

    for sub in sorted(CLEAN_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("nm"):
            continue
        lam_nm = int(sub.name[:-2])
        stack = np.load(sub / "stack_clean.npy").astype(np.float32)
        z_um  = np.load(sub / "z_um.npy").astype(np.float32)

        s1 = sharpness_curve(stack, "lapvar")
        s2 = sharpness_curve(stack, "grad")
        s3 = sharpness_curve(stack, "fft")

        i1 = int(np.argmax(s1)); z1 = float(z_um[i1])
        i2 = int(np.argmax(s2)); z2 = float(z_um[i2])
        i3 = int(np.argmax(s3)); z3 = float(z_um[i3])

        # консенсус — медиана трёх оценок по z
        z_cons = float(np.median([z1, z2, z3]))

        rows.append({
            "lambda_nm": lam_nm,
            "best_lapvar_z_um": z1,
            "best_grad_z_um":   z2,
            "best_fft_z_um":    z3,
            "consensus_z_um":   z_cons
        })

        # График
        plt.figure()
        plt.title(f"Sharpness vs z — {lam_nm} nm")
        plt.plot(z_um, s1, label="Laplacian var")
        plt.plot(z_um, s2, label="Gradient energy")
        plt.plot(z_um, s3, label="High‑freq FFT")
        plt.axvline(z_cons, ls="--", label=f"consensus {z_cons:.2f} μm")
        plt.xlabel("z (μm)"); plt.ylabel("sharpness"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{lam_nm}nm_sharpness.png", dpi=150)
        plt.close()

    df = pd.DataFrame(rows).sort_values("lambda_nm").reset_index(drop=True)
    df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"Готово: {OUT_DIR / 'summary.csv'}")

if __name__ == "__main__":
    main()
