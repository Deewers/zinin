# -*- coding: utf-8 -*-
"""
Шаг 3A (SIM). Восстановление поля методом Gerchberg–Saxton вокруг лучшего фокуса.
Берём три плоскости: z* и соседние (z*-Δz, z*+Δz). Итеративно восстанавливаем U0(x,y;λ).

Вход:  out/sim/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
Выход: out/phase_gs/<λ>nm/{U0.npy, amplitude.png, phase.png, Pupil.npy, pupil_*.png}
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
CLEAN_DIR = Path("out")/"sim"          # работаем только с сим-данными
OUT_DIR   = Path("out")/"phase_gs"

# ---------- Утилиты ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_sim():
    data = {}
    if not CLEAN_DIR.is_dir():
        print(f"⚠️ Нет каталога {CLEAN_DIR}")
        return data
    for sub in sorted(CLEAN_DIR.iterdir()):
        if sub.is_dir() and sub.name.endswith("nm"):
            try:
                lam = int(sub.name[:-2])
            except:
                continue
            stack = np.load(sub/"stack_clean.npy")      # [Z,H,W]
            z_um  = np.load(sub/"z_um.npy").astype(float)
            idx   = int(np.load(sub/"idx_best.npy")[0])
            data[lam] = {"stack": stack, "z_um": z_um, "idx_best": idx}
    return data

class AngularSpectrum2D:
    def __init__(self, lam_m: float, dx_m: float, ny: int, nx: int, na: float):
        self.lam=lam_m; self.dx=dx_m; self.ny, self.nx=ny, nx
        fy = np.fft.fftfreq(ny, d=dx_m)
        fx = np.fft.fftfreq(nx, d=dx_m)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing='xy')
        self.F2 = self.FX**2 + self.FY**2
        self.k  = 2*np.pi/self.lam
        self.band = (self.F2 <= (na/self.lam)**2).astype(np.float32)
        arg = np.maximum(1.0 - (self.lam**2)*self.F2, 0.0)
        self.kz = self.k*np.sqrt(arg)
    def H(self, z_m: float):
        return np.exp(1j*self.kz*z_m)*self.band

def normalize01(a: np.ndarray)->np.ndarray:
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (1.0, 99.5))
    if hi-lo < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if hi-lo < 1e-12: return np.zeros_like(a)
    a = (a-lo)/(hi-lo)
    return np.clip(a,0,1)

def gs_retrieve(I_stack: np.ndarray, z_um: np.ndarray, lam_nm: int, dx_um: float, na: float,
                idx_center: int, n_iters: int = 30):
    assert I_stack.ndim == 3 and I_stack.shape[0] >= 3
    I0 = I_stack[idx_center].astype(np.float64)

    idx_m1 = idx_center-1
    idx_p1 = idx_center+1
    if idx_m1 < 0 or idx_p1 >= I_stack.shape[0]:
        raise ValueError("Недостаточно соседних плоскостей для GS.")
    I_minus = I_stack[idx_m1].astype(np.float64)
    I_plus  = I_stack[idx_p1].astype(np.float64)
    dz_um   = float(z_um[idx_p1] - z_um[idx_center])  # шаг симуляции (д.б. 10 µm)

    lam_m = lam_nm * 1e-9
    dx_m  = dx_um * 1e-6
    ny, nx = I0.shape
    asp = AngularSpectrum2D(lam_m, dx_m, ny, nx, na)

    # начальная оценка
    U0 = np.sqrt(I0).astype(np.complex128)

    H_p = asp.H(dz_um*1e-6)
    H_m = asp.H(-dz_um*1e-6)

    for _ in range(n_iters):
        # -> z+
        U_plus = np.fft.ifft2(np.fft.fft2(U0)*H_p)
        U_plus = np.sqrt(I_plus)*np.exp(1j*np.angle(U_plus))
        # -> обратно в 0
        U0 = np.fft.ifft2(np.fft.fft2(U_plus)*H_m)
        # -> z-
        U_minus = np.fft.ifft2(np.fft.fft2(U0)*H_m)
        U_minus = np.sqrt(I_minus)*np.exp(1j*np.angle(U_minus))
        # -> обратно в 0
        U0 = np.fft.ifft2(np.fft.fft2(U_minus)*H_p)
        # проекция амплитуды в 0
        U0 = np.sqrt(I0)*np.exp(1j*np.angle(U0))

    return U0.astype(np.complex64)

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в out/config.json.")
    ensure_dir(OUT_DIR)
    all_data = load_sim()
    for lam in sorted(all_data.keys()):
        d = all_data[lam]
        stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        if stack.shape[0] < 3:
            print(f"λ={lam} nm: слоёв меньше 3 → пропуск")
            continue
        NA = na_for_lambda(lam)
        U0 = gs_retrieve(stack, z_um, lam, DX_UM, NA, idx_center=idx, n_iters=30)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out/"U0.npy", U0)

        amp = normalize01(np.abs(U0))
        ph  = np.angle(U0).astype(np.float32)

        plt.figure(); plt.title(f"Amplitude — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out/"amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Phase — {lam} nm"); plt.imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil (жёсткая апертура)
        lam_m = lam*1e-9; dx_m = DX_UM*1e-6
        ny, nx = U0.shape
        fx = np.fft.fftfreq(nx, d=dx_m); fy = np.fft.fftfreq(ny, d=dx_m)
        FX,FY = np.meshgrid(fx, fy, indexing='xy')
        band = (FX**2 + FY**2 <= (NA/lam_m)**2)
        F = np.fft.fft2(U0)
        P = F * band
        np.save(out/"Pupil.npy", P.astype(np.complex64))

        Pamp = normalize01(np.abs(P))
        Pph  = np.angle(P).astype(np.float32)
        plt.figure(); plt.title(f"Pupil amplitude — {lam} nm"); plt.imshow(np.fft.fftshift(Pamp), cmap="gray"); plt.axis("off")
        plt.savefig(out/"pupil_amplitude.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); plt.title(f"Pupil phase — {lam} nm"); plt.imshow(np.fft.fftshift(Pph), cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"pupil_phase.png", dpi=150, bbox_inches="tight"); plt.close()

        print(f"λ={lam} nm → {out}")

if __name__ == "__main__":
    main()
