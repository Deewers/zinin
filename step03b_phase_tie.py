# -*- coding: utf-8 -*-
"""
Шаг 3B. Восстановление фазы методом TIE (Transport of Intensity Equation).
Берём I(z*±Δz) и I(z*). Решаем лапласиан фазы в частотной области (параксиально).

Вход:  out/clean/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
Выход: out/phase_tie/<λ>nm/{phi_tie.npy, U0.npy, phase.png, amplitude.png, Pupil.npy, pupil_*.png}
"""

from pathlib import Path
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- ПАРАМЕТРЫ / КОНФИГ ----
CFG = Path("out") / "config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))

def na_for_lambda(l):
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(l), cfg.get("na_default", 0.25)))

EPS_I0 = 1e-6  # защита от деления на малые интенсивности

from pathlib import Path
# по умолчанию работаем с out/sim, если он есть; если хотите — верните на out/clean
CLEAN_DIR = Path("out")/"sim"
if not CLEAN_DIR.is_dir():
    CLEAN_DIR = Path("out")/"clean_aligned" if (Path("out")/"clean_aligned").is_dir() else Path("out")/"clean"


OUT_DIR   = Path("out") / "phase_tie"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

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

def solve_poisson_fft(rhs: np.ndarray, dx_m: float) -> np.ndarray:
    """Решение ∇²φ = rhs в Фурье-области (DC=0)."""
    ny, nx = rhs.shape
    fx = np.fft.fftfreq(nx, d=dx_m)
    fy = np.fft.fftfreq(ny, d=dx_m)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    denom = - (2*np.pi)**2 * (FX**2 + FY**2)
    RHS = np.fft.fft2(rhs)
    denom[0,0] = 1.0  # чтобы не делить на ноль
    PHI = RHS / denom
    PHI[0,0] = 0.0
    phi = np.real(np.fft.ifft2(PHI))
    return phi

def tie_phase(I0: np.ndarray, Iplus: np.ndarray, Iminus: np.ndarray, dz_m: float, lam_m: float, dx_m: float) -> np.ndarray:
    """Классическая TIE с предположением медленного изменения I0: ∇²φ ≈ -(2π/(λ I0)) ∂I/∂z"""
    dIdz = (Iplus - Iminus) / (2.0 * dz_m)
    I0c = np.maximum(I0, EPS_I0)
    rhs = - (2*np.pi/lam_m) * (dIdz / I0c)  # правая часть уравнения на ∇²φ
    phi = solve_poisson_fft(rhs, dx_m)
    return phi

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в out/config.json или в шапке скрипта.")
    ensure_dir(OUT_DIR)
    data = load_clean()
    for lam in sorted(data.keys()):
        d = data[lam]
        stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        if idx == 0 or idx == stack.shape[0]-1:
            print(f"λ={lam} nm: лучший индекс на краю, TIE нужен сосед с обеих сторон → пропуск")
            continue

        # --- ВАЖНО: NA для этой длины волны ---
        NA = na_for_lambda(lam)

        I0 = stack[idx].astype(np.float64)
        Iplus  = stack[idx+1].astype(np.float64)
        Iminus = stack[idx-1].astype(np.float64)
        dz_um  = float(z_um[idx+1] - z_um[idx])
        lam_m  = lam * 1e-9
        dx_m   = DX_UM * 1e-6
        dz_m   = dz_um * 1e-6

        phi = tie_phase(I0, Iplus, Iminus, dz_m, lam_m, dx_m)
        U0 = np.sqrt(I0) * np.exp(1j*phi)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out / "phi_tie.npy", phi.astype(np.float32))
        np.save(out / "U0.npy", U0.astype(np.complex64))

        amp = np.abs(U0); amp /= (amp.max() + 1e-12)
        plt.figure(); plt.title(f"Amplitude (TIE) — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out / "amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Phase (TIE) — {lam} nm"); plt.imshow(phi, cmap="gray"); plt.axis("off")
        plt.savefig(out / "phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil (жёсткая апертура)
        F = np.fft.fft2(U0)
        fx = np.fft.fftfreq(U0.shape[1], d=dx_m)
        fy = np.fft.fftfreq(U0.shape[0], d=dx_m)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        band = (FX**2 + FY**2 <= (NA/lam_m)**2)
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

        print(f"λ={lam} nm (NA={NA:.3f}) → {out}")

if __name__ == "__main__":
    main()
