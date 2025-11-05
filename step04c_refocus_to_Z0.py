# -*- coding: utf-8 -*-
"""
Шаг 4C. Виртуальный рефокус всех каналов к общей плоскости Z0.
Z0 берём из out/focus/summary.csv (consensus медиана), либо задаём вручную.

Вход:  out/focus/summary.csv,
       out/clean/<λ>nm/{z_um.npy, idx_best.npy},
       best available field: out/phase_gs/<λ>nm/U0.npy  ИЛИ
                            out/phase_tie/<λ>nm/U0.npy  ИЛИ
                            out/phase_analytic/<λ>nm/U_focus_analytic.npy
Выход: out/refocus/<method>/<λ>nm/{U_ref.npy, I_ref.png}
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ
DX_UM = 3.3  # микрон/пиксель — укажите ваш масштаб!
NA     = 0.25
Z0_UM: float | None = None  # если None — берём медиану consensus из out/focus/summary.csv

FOCUS_SUMMARY = Path("out") / "focus" / "summary.csv"
CLEAN_DIR     = Path("out") / "clean"
PHASE_DIRS    = [
    ("phase_gs",       Path("out") / "phase_gs",       "U0.npy"),
    ("phase_tie",      Path("out") / "phase_tie",      "U0.npy"),
    ("phase_analytic", Path("out") / "phase_analytic", "U_focus_analytic.npy"),
]
OUT_DIR       = Path("out") / "refocus"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

class AngularSpectrum2D:
    def __init__(self, lam_m: float, dx_m: float, n: int, na: float):
        self.lam=lam_m; self.dx=dx_m; self.n=n
        fx = np.fft.fftfreq(n, d=dx_m)
        fy = np.fft.fftfreq(n, d=dx_m)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing='xy')
        self.F2 = self.FX**2 + self.FY**2
        self.k  = 2*np.pi/self.lam
        self.band = (self.F2 <= (na/self.lam)**2).astype(np.float32)
    def H(self, z_m: float):
        arg = np.maximum(1.0 - (self.lam**2)*self.F2, 0.0)
        kz  = self.k*np.sqrt(arg)
        return np.exp(1j*kz*z_m)*self.band
    def propagate(self, U0: np.ndarray, z_m: float) -> np.ndarray:
        F = np.fft.fft2(U0) * self.band
        return np.fft.ifft2(F * self.H(z_m))

def pick_field(lam_nm: int) -> tuple[np.ndarray, str] | tuple[None, None]:
    for tag, root, fname in PHASE_DIRS:
        p = root / f"{lam_nm}nm" / fname
        if p.exists():
            return np.load(p), tag
    return None, None

def load_z_best(lam_nm: int) -> float:
    z = np.load(CLEAN_DIR / f"{lam_nm}nm" / "z_um.npy").astype(float)
    idx = int(np.load(CLEAN_DIR / f"{lam_nm}nm" / "idx_best.npy")[0])
    return float(z[idx])

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в начале файла.")
    ensure_dir(OUT_DIR)

    # Z0
    z0_um = Z0_UM
    if z0_um is None:
        if not FOCUS_SUMMARY.exists():
            raise SystemExit("Нет out/focus/summary.csv. Сначала запустите step04a_focus_summary.py")
        df = pd.read_csv(FOCUS_SUMMARY)
        z0_um = float(np.median(df["consensus_z_um"].values))
    print(f"Целевая плоскость Z0 = {z0_um:.3f} μm")

    # Проходим все λ
    for sub in sorted(CLEAN_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("nm"):
            continue
        lam_nm = int(sub.name[:-2])
        U, tag = pick_field(lam_nm)
        if U is None:
            print(f"λ={lam_nm} nm: нет комплексного поля в phase_* → пропуск")
            continue

        lam_m = lam_nm*1e-9; dx_m = DX_UM*1e-6
        z_best = load_z_best(lam_nm)
        dz_m   = (z0_um - z_best)*1e-6

        asp = AngularSpectrum2D(lam_m, dx_m, U.shape[0], NA)
        Uref = asp.propagate(U, dz_m)
        Iref = np.abs(Uref)**2
        if Iref.max()>0: Iref /= Iref.max()

        out = OUT_DIR / tag / f"{lam_nm}nm"
        ensure_dir(out)
        np.save(out / "U_ref.npy",  Uref.astype(np.complex64))
        np.save(out / "I_ref.npy",  Iref.astype(np.float32))

        plt.figure(); plt.title(f"Refocused to Z0={z0_um:.2f} μm — {lam_nm} nm ({tag})")
        plt.imshow(Iref, cmap="gray"); plt.axis("off")
        plt.tight_layout(); plt.savefig(out / "I_ref.png", dpi=150); plt.close()

        print(f"λ={lam_nm} nm ({tag}): перенесено на ΔZ={z0_um-z_best:+.2f} μm → {out}")

if __name__ == "__main__":
    main()
