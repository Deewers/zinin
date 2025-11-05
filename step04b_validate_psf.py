# -*- coding: utf-8 -*-
"""
Шаг 4B. Валидация модели оптики по "Мире".
- Берём лучший кадр как R0 (приближённый объект).
- По Pupil(λ) синтезируем PSF(Δz) и предсказываем I_pred = R0 ⊗ PSF(Δz) (инкоherent-приближение).
- Сравниваем I_pred с измеренным I_meas(z): корреляция (r), MSE, графики.
Вход:  out/clean/<λ>nm/{stack_clean.npy, z_um.npy}, Pupil из out/phase_*/*/Pupil.npy
Выход: out/validate/<λ>nm/{validation.csv, psf_best.png, compare_z=..png, ...}
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
K_AROUND = 5   # сколько плоскостей с каждой стороны от лучшей проверять

CLEAN_DIR      = Path("out") / "clean"
PHASE_DIR_GS   = Path("out") / "phase_gs"
PHASE_DIR_TIE  = Path("out") / "phase_tie"
PHASE_DIR_ANL  = Path("out") / "phase_analytic"
OUT_DIR        = Path("out") / "validate"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def pick_pupil(lam_nm: int):
    for root in [PHASE_DIR_GS, PHASE_DIR_TIE, PHASE_DIR_ANL]:
        p = root / f"{lam_nm}nm" / "Pupil.npy"
        if p.exists():
            return np.load(p), root.name
    return None, None

class AngularSpectrum2D:
    def __init__(self, lam_m: float, dx_m: float, n: int, na: float):
        self.lam = lam_m; self.dx = dx_m; self.n=n
        fx = np.fft.fftfreq(n, d=dx_m)
        fy = np.fft.fftfreq(n, d=dx_m)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing='xy')
        self.F2 = self.FX**2 + self.FY**2
        self.k  = 2*np.pi/self.lam
        self.band = (self.F2 <= (na/self.lam)**2).astype(np.float32)
    def H(self, z_m: float):
        arg = np.maximum(1.0 - (self.lam**2)*self.F2, 0.0)
        kz = self.k*np.sqrt(arg)
        return np.exp(1j*kz*z_m)*self.band

def synthesize_psf_from_pupil(P: np.ndarray, z_m: float, lam_m: float, dx_m: float, na: float) -> np.ndarray:
    asp = AngularSpectrum2D(lam_m, dx_m, P.shape[0], na)
    Uz = np.fft.ifft2(P * asp.H(z_m))
    I  = np.abs(Uz)**2
    I /= (I.max() + 1e-12)
    return I

def fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    return np.real(np.fft.ifft2(A*B))

def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    da = np.sqrt((a*a).mean()) + 1e-12
    db = np.sqrt((b*b).mean()) + 1e-12
    return float((a*b).mean()/(da*db))

def validate_lambda(lam_nm: int, stack: np.ndarray, z_um: np.ndarray, idx_best: int, P: np.ndarray, out_dir: Path):
    ensure_dir(out_dir)
    lam_m = lam_nm*1e-9; dx_m = DX_UM*1e-6
    R0 = stack[idx_best]  # приблизительный "объект"
    rows = []
    z0 = float(z_um[idx_best])
    z_indices = range(max(0, idx_best-K_AROUND), min(len(z_um), idx_best+K_AROUND+1))
    for k in z_indices:
        dz_m = (float(z_um[k]) - z0)*1e-6
        psf = synthesize_psf_from_pupil(P, dz_m, lam_m, dx_m, NA)
        pred = fft_convolve2d(R0, psf)
        if pred.max()>0: pred /= pred.max()
        meas = stack[k]
        r = pearsonr(pred, meas)
        mse = float(((pred - meas)**2).mean())
        rows.append({"z_um": float(z_um[k]), "dz_um": float(z_um[k]-z0), "pearson_r": r, "mse": mse})

        # сохраните несколько сравнений
        if k in [idx_best, min(len(z_um)-1, idx_best+K_AROUND), max(0, idx_best-K_AROUND)]:
            fig = plt.figure(figsize=(9,3))
            plt.suptitle(f"{lam_nm} nm — z={z_um[k]:.2f} μm (dz={z_um[k]-z0:.2f})")
            plt.subplot(1,3,1); plt.title("measured"); plt.imshow(meas, cmap="gray"); plt.axis("off")
            plt.subplot(1,3,2); plt.title("predicted"); plt.imshow(pred, cmap="gray"); plt.axis("off")
            plt.subplot(1,3,3); plt.title(f"diff (MSE={mse:.3g})"); plt.imshow(np.abs(pred-meas), cmap="gray"); plt.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"compare_z={z_um[k]:+.2f}.png", dpi=150)
            plt.close(fig)

    df = pd.DataFrame(rows).sort_values("dz_um")
    df.to_csv(out_dir / "validation.csv", index=False, encoding="utf-8-sig")

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в начале файла.")
    ensure_dir(OUT_DIR)

    # проходим по out/clean
    for sub in sorted(CLEAN_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("nm"):
            continue
        lam_nm = int(sub.name[:-2])
        stack = np.load(sub / "stack_clean.npy").astype(np.float32)
        z_um  = np.load(sub / "z_um.npy").astype(np.float32)
        # idx_best (если нет, берём максимум яркости)
        idx_path = sub / "idx_best.npy"
        if idx_path.exists():
            idx_best = int(np.load(idx_path)[0])
        else:
            idx_best = int(np.argmax(stack.reshape(stack.shape[0], -1).sum(axis=1)))

        P, src = pick_pupil(lam_nm)
        if P is None:
            print(f"λ={lam_nm} nm: нет Pupil в phase_gs/phase_tie/phase_analytic → пропуск")
            continue

        lam_dir = OUT_DIR / f"{lam_nm}nm"
        validate_lambda(lam_nm, stack, z_um, idx_best, P, lam_dir)
        print(f"λ={lam_nm} nm: валидация выполнена (pupil from {src}) → {lam_dir}")

if __name__ == "__main__":
    main()
