# -*- coding: utf-8 -*-
"""
Шаг 3C (SIM). Аналитический рефокус: перенос каждой плоскости в фокус и кохерентное усреднение.
U_focus = mean_k( IFFT( FFT(Uk) * H(-Δz_k) ) ), где Uk = sqrt(Ik)*exp(i*phi0 на центральной плоскости, иначе фаза=0)

Вход:  out/sim/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
      (+опц.) out/phase_tie/<λ>nm/phi_tie.npy — используем как φ0 в центральной плоскости
Выход: out/phase_analytic/<λ>nm/{U_focus_analytic.npy, amplitude.png, phase.png, Pupil.npy, pupil_*.png}
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
TIE_DIR   = Path("out")/"phase_tie"
OUT_DIR   = Path("out")/"phase_analytic"

# ---------- Утилиты ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_sim():
    data = {}
    if not CLEAN_DIR.is_dir(): return data
    for sub in sorted(CLEAN_DIR.iterdir()):
        if sub.is_dir() and sub.name.endswith("nm"):
            try: lam = int(sub.name[:-2])
            except: continue
            stack = np.load(sub/"stack_clean.npy")
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

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (µm/px) в out/config.json.")
    ensure_dir(OUT_DIR)
    data = load_sim()
    for lam in sorted(data.keys()):
        d = data[lam]; stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        lam_m = lam * 1e-9; dx_m = DX_UM * 1e-6
        NA = na_for_lambda(lam)
        ny, nx = stack.shape[1], stack.shape[2]
        asp = AngularSpectrum2D(lam_m, dx_m, ny, nx, NA)

        # фаза в центре z*: если есть TIE — используем её
        phi_path = TIE_DIR/f"{lam}nm"/"phi_tie.npy"
        phi0 = np.load(phi_path).astype(np.float64) if phi_path.exists() else None

        U_acc = None
        for k in range(stack.shape[0]):
            Ik = stack[k].astype(np.float64)
            amp = np.sqrt(Ik)
            if phi0 is not None and k == idx:
                Uk = amp * np.exp(1j*phi0)  # в центре используем TIE фазу
            else:
                Uk = amp.astype(np.complex128)  # иначе фаза 0

            dz_m = (z_um[k] - z_um[idx]) * 1e-6
            # перенос в фокус (на -dz)
            U0_k = np.fft.ifft2(np.fft.fft2(Uk) * asp.H(-dz_m))
            U_acc = U0_k if U_acc is None else (U_acc + U0_k)

        U_focus = (U_acc / float(stack.shape[0])).astype(np.complex64)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out/"U_focus_analytic.npy", U_focus)

        amp = normalize01(np.abs(U_focus))
        ph  = np.angle(U_focus).astype(np.float32)
        plt.figure(); plt.title(f"Amplitude (analytic) — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out/"amplitude.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); plt.title(f"Phase (analytic) — {lam} nm"); plt.imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil (жёсткая апертура)
        ny, nx = U_focus.shape
        fx = np.fft.fftfreq(nx, d=dx_m); fy = np.fft.fftfreq(ny, d=dx_m)
        FX,FY = np.meshgrid(fx, fy, indexing='xy')
        band = (FX**2 + FY**2 <= (NA/lam_m)**2)
        P = np.fft.fft2(U_focus) * band
        np.save(out/"Pupil.npy", P.astype(np.complex64))

        Pamp = normalize01(np.abs(P)); Pph = np.angle(P)
        plt.figure(); plt.title(f"Pupil amplitude — {lam} nm"); plt.imshow(np.fft.fftshift(Pamp), cmap="gray"); plt.axis("off")
        plt.savefig(out/"pupil_amplitude.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); plt.title(f"Pupil phase — {lam} nm"); plt.imshow(np.fft.fftshift(Pph), cmap="twilight", vmin=-np.pi, vmax=np.pi); plt.axis("off")
        plt.savefig(out/"pupil_phase.png", dpi=150, bbox_inches="tight"); plt.close()

        print(f"λ={lam} nm → {out}")

if __name__ == "__main__":
    main()
