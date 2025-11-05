# -*- coding: utf-8 -*-
"""
Шаг 3B (SIM). Восстановление фазы методом TIE (Transport of Intensity Equation).
Берём I(z*±Δz) и I(z*). Решаем ∇²φ = -(2π/λ) * (1/I0) * ∂I/∂z в частотной области.

Вход:  out/sim/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
Выход: out/phase_tie/<λ>nm/{phi_tie.npy, U0.npy, amplitude.png, phase.png, Pupil.npy, pupil_*.png}
"""

from pathlib import Path
import json, numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Конфиг ----------
CFG = Path("out")/"config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))
def na_for_lambda(lam_nm:int)->float:
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(lam_nm), cfg.get("na_default", 0.25)))

# ---------- Директории ----------
CLEAN_DIR = Path("out")/"sim"
OUT_DIR   = Path("out")/"phase_tie"

# ---------- Утилиты ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_sim():
    data = {}
    if not CLEAN_DIR.is_dir(): return data
    for sub in sorted(CLEAN_DIR.iterdir()):
        if sub.is_dir() and sub.name.endswith("nm"):
            try:
                lam = int(sub.name[:-2])
            except: continue
            stack = np.load(sub/"stack_clean.npy")
            z_um  = np.load(sub/"z_um.npy").astype(float)
            idx   = int(np.load(sub/"idx_best.npy")[0])
            data[lam] = {"stack": stack, "z_um": z_um, "idx_best": idx}
    return data

def normalize01(a: np.ndarray)->np.ndarray:
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (1.0, 99.5))
    if hi-lo < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if hi-lo < 1e-12: return np.zeros_like(a)
    a = (a-lo)/(hi-lo)
    return np.clip(a,0,1)

def solve_poisson_fft(rhs: np.ndarray, dx_m: float) -> np.ndarray:
    ny, nx = rhs.shape
    fx = np.fft.fftfreq(nx, d=dx_m)
    fy = np.fft.fftfreq(ny, d=dx_m)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    denom = - (2*np.pi)**2 * (FX**2 + FY**2)
    RHS = np.fft.fft2(rhs)
    denom[0,0] = 1.0
    PHI = RHS / denom
    PHI[0,0] = 0.0
    phi = np.real(np.fft.ifft2(PHI))
    return phi

def tie_phase(I0: np.ndarray, Iplus: np.ndarray, Iminus: np.ndarray, dz_m: float, lam_m: float, dx_m: float) -> np.ndarray:
    EPS_I0 = 1e-6
    dIdz = (Iplus - Iminus) / (2.0 * dz_m)
    I0c  = np.maximum(I0, EPS_I0)
    rhs  = - (2*np.pi/lam_m) * (dIdz / I0c)
    phi  = solve_poisson_fft(rhs, dx_m)
    return phi

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (µm/px) в out/config.json.")
    ensure_dir(OUT_DIR)
    data = load_sim()
    for lam in sorted(data.keys()):
        d = data[lam]
        stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        if idx == 0 or idx == stack.shape[0]-1:
            print(f"λ={lam} nm: лучший индекс на краю (TIE нужен сосед с обеих сторон) → пропуск")
            continue
        I0     = stack[idx].astype(np.float64)
        Iplus  = stack[idx+1].astype(np.float64)
        Iminus = stack[idx-1].astype(np.float64)
        dz_um  = float(z_um[idx+1] - z_um[idx])
        lam_m  = lam * 1e-9
        dx_m   = DX_UM * 1e-6
        dz_m   = dz_um * 1e-6

        phi = tie_phase(I0, Iplus, Iminus, dz_m, lam_m, dx_m)
        U0  = np.sqrt(I0) * np.exp(1j*phi)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out/"phi_tie.npy", phi.astype(np.float32))
        np.save(out/"U0.npy",     U0.astype(np.complex64))

        amp = normalize01(np.abs(U0))
        plt.figure(); plt.title(f"Amplitude (TIE) — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out/"amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Phase (TIE) — {lam} nm"); plt.imshow(np.angle(U0), cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil (жёсткая апертура)
        NA    = na_for_lambda(lam)
        lam_m = lam*1e-9; dx_m = DX_UM*1e-6
        ny, nx = U0.shape
        fx = np.fft.fftfreq(nx, d=dx_m); fy = np.fft.fftfreq(ny, d=dx_m)
        FX,FY = np.meshgrid(fx, fy, indexing='xy')
        band = (FX**2 + FY**2 <= (NA/lam_m)**2)
        P = np.fft.fft2(U0) * band
        np.save(out/"Pupil.npy", P.astype(np.complex64))

        Pamp = normalize01(np.abs(P)); Pph = np.angle(P)
        plt.figure(); plt.title(f"Pupil amplitude — {lam} nm"); plt.imshow(np.fft.fftshift(Pamp), cmap="gray"); plt.axis("off")
        plt.savefig(out/"pupil_amplitude.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); plt.title(f"Pupil phase — {lam} nm"); plt.imshow(np.fft.fftshift(Pph), cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"pupil_phase.png", dpi=150, bbox_inches="tight"); plt.close()

        print(f"λ={lam} nm → {out}")

if __name__ == "__main__":
    main()
