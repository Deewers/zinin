# -*- coding: utf-8 -*-
"""
Шаг 3A. Восстановление фазы/поля методом Gerchberg–Saxton вокруг лучшего фокуса.
Берём три плоскости: z* и соседние (z*-Δz, z*+Δz). Итеративно восстанавливаем U0(x,y;λ).

Вход:  out/clean/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
Выход: out/phase_gs/<λ>nm/{U0.npy, amplitude.png, phase.png, Pupil.npy, pupil_*.png, psf_synth.npy}
"""

import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- ПАРАМЕТРЫ: ОБЯЗАТЕЛЬНО ВВЕДИТЕ СВОИ ----
# Чтение DX и NA из конфигурационного файла
from pathlib import Path
import json

# Чтение конфигурации из файла
CFG = Path("out") / "config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}

# Чтение DX (шаг пикселя) из конфигурации
DX_UM = float(cfg.get("dx_um", 3.3))  # по умолчанию 3.3 µm/px

# Функция для получения NA для данной длины волны
def na_for_lambda(l):
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(l), cfg.get("na_default", 0.25)))  # если нет, то по дефолту 0.25

# Внутри цикла по λ: используйте:
# NA = na_for_lambda(lam_nm)

def gs_retrieve(I_stack: np.ndarray, z_um: np.ndarray, lam_nm: float, dx_um: float, na: float,
                idx_center: int, n_iters: int = 30):
    lam_m = lam_nm * 1e-9
    dx_m = dx_um * 1e-6
    # Используем NA, полученное из конфигурации
    asp = AngularSpectrum2D(lam_m, dx_m, I_stack.shape[0], na)

N_ITERS = 30   # число итераций GS

from pathlib import Path
# по умолчанию работаем с out/sim, если он есть; если хотите — верните на out/clean
CLEAN_DIR = Path("out")/"sim"
if not CLEAN_DIR.is_dir():
    CLEAN_DIR = Path("out")/"clean_aligned" if (Path("out")/"clean_aligned").is_dir() else Path("out")/"clean"

CLEAN_ALIGNED = Path("out") / "clean_aligned"
if CLEAN_ALIGNED.is_dir():
    print("✅ Найден выровненный стек → использую out/clean_aligned")
    CLEAN_DIR = CLEAN_ALIGNED

OUT_DIR   = Path("out") / "phase_gs"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_clean():
    data = {}
    if not CLEAN_DIR.is_dir():
        print(f"⚠️ Нет каталога {CLEAN_DIR}")
        return data
    for sub in sorted(CLEAN_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("nm"):
            continue
        lam_nm = int(sub.name[:-2])
        stack = np.load(sub / "stack_clean.npy")      # [Z,H,W]
        z_um  = np.load(sub / "z_um.npy").astype(float)
        idx   = int(np.load(sub / "idx_best.npy")[0])
        data[lam_nm] = {"stack": stack, "z_um": z_um, "idx_best": idx}
    return data

class AngularSpectrum2D:
    def __init__(self, wavelength_m: float, dx_m: float, nx: int, na: float):
        self.lam = wavelength_m
        self.dx  = dx_m
        self.n   = nx
        fx = np.fft.fftfreq(nx, d=dx_m)
        fy = np.fft.fftfreq(nx, d=dx_m)
        self.fx, self.fy = np.meshgrid(fx, fy, indexing='xy')
        self.f2 = self.fx**2 + self.fy**2
        self.k = 2*np.pi/self.lam
        f_cut = na / self.lam  # cycles/m
        self.band = (self.f2 <= f_cut**2).astype(np.float32)

    def H(self, z_m: float):
        arg = 1.0 - (self.lam**2)*self.f2
        arg = np.maximum(arg, 0.0)
        kz  = self.k * np.sqrt(arg)
        H = np.exp(1j*kz*z_m) * self.band
        return H

def gs_retrieve(I_stack: np.ndarray, z_um: np.ndarray, lam_nm: float, dx_um: float, na: float,
                idx_center: int, n_iters: int = 30):
    assert I_stack.ndim == 3 and I_stack.shape[0] >= 3
    z = z_um
    I0     = I_stack[idx_center].astype(np.float64)
    idx_m1 = max(0, idx_center-1)
    idx_p1 = min(I_stack.shape[0]-1, idx_center+1)
    if idx_m1 == idx_center: idx_m1 = idx_center-1
    if idx_p1 == idx_center: idx_p1 = idx_center+1

    I_minus = I_stack[idx_m1].astype(np.float64)
    I_plus  = I_stack[idx_p1].astype(np.float64)
    dz_um   = float(z[idx_p1] - z[idx_center])  # считаем симметрию
    lam_m   = lam_nm * 1e-9
    dx_m    = dx_um * 1e-6

    asp = AngularSpectrum2D(lam_m, dx_m, I0.shape[0], na)

    U0 = np.sqrt(I0).astype(np.complex128)  # нулевая фаза

    H_p = asp.H(dz_um*1e-6)
    H_m = asp.H(-dz_um*1e-6)

    for _ in range(n_iters):
        # -> z+
        F = np.fft.fft2(U0) * asp.band
        U_plus = np.fft.ifft2(F * H_p)
        U_plus = np.sqrt(I_plus) * np.exp(1j*np.angle(U_plus))
        # -> обратно в 0
        F = np.fft.fft2(U_plus) * asp.band
        U0 = np.fft.ifft2(F * H_m)
        # -> z-
        F = np.fft.fft2(U0) * asp.band
        U_minus = np.fft.ifft2(F * H_m)
        U_minus = np.sqrt(I_minus) * np.exp(1j*np.angle(U_minus))
        # -> обратно в 0
        F = np.fft.fft2(U_minus) * asp.band
        U0 = np.fft.ifft2(F * H_p)
        # проекция амплитуды в 0
        U0 = np.sqrt(I0) * np.exp(1j*np.angle(U0))

    return U0.astype(np.complex64), dz_um

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в начале файла.")
    ensure_dir(OUT_DIR)
    all_data = load_clean()
    for lam in sorted(all_data.keys()):
        d = all_data[lam]
        stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        if stack.shape[0] < 3:
            print(f"λ={lam} nm: слишком мало слоёв (нужно >=3)")
            continue
        NA = na_for_lambda(lam)  # используем NA для текущей λ
        U0, dz_um = gs_retrieve(stack, z_um, lam, DX_UM, NA, idx_center=idx, n_iters=N_ITERS)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out / "U0.npy", U0)

        amp = np.abs(U0); amp /= (amp.max() + 1e-12)
        ph  = np.angle(U0)

        plt.figure(); plt.title(f"Amplitude — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out / "amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Phase — {lam} nm"); plt.imshow(ph, cmap="gray"); plt.axis("off")
        plt.savefig(out / "phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil
        F = np.fft.fft2(U0)
        # жёсткая апертура
        dx_m = DX_UM*1e-6; lam_m = lam*1e-9
        fx = np.fft.fftfreq(U0.shape[0], d=dx_m); fy = np.fft.fftfreq(U0.shape[0], d=dx_m)
        FX, FY = np.meshgrid(fx, fy, indexing='xy'); band = (FX**2 + FY**2 <= (NA/lam_m)**2)
        P = F * band
        np.save(out / "Pupil.npy", P.astype(np.complex64))

        Pamp = np.abs(P); Pamp /= (Pamp.max() + 1e-12)
        Pph  = np.angle(P)

        plt.figure(); plt.title(f"Pupil amplitude — {lam} nm")
        plt.imshow(np.fft.fftshift(Pamp), cmap="gray"); plt.axis("off")
        plt.savefig(out / "pupil_amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Pupil phase — {lam} nm")
        plt.imshow(np.fft.fftshift(Pph), cmap="gray"); plt.axis("off")
        plt.savefig(out / "pupil_phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Синтетический PSF по нескольким z для проверки
        zlist = z_um
        psf_syn = []
        f0 = P
        for zz in zlist:
            # угловой спектр
            asp = AngularSpectrum2D(lam_m, dx_m, U0.shape[0], NA)
            Uz = np.fft.ifft2(f0 * asp.H((zz - z_um[idx])*1e-6))
            psf_syn.append(np.abs(Uz)**2)
        np.save(out / "psf_synth.npy", np.stack(psf_syn, axis=0).astype(np.float32))
        print(f"λ={lam} nm → {out}")

if __name__ == "__main__":
    main()
