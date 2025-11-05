# -*- coding: utf-8 -*-
"""
Шаг 3C. Аналитический рефокус по параксиальной фазе дефокуса (без итераций).
Û_corr = Û_Z * exp(+ i π λ ΔZ (u^2+v^2)) — обратный перенос в фокус.

Вход:  out/clean/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
      (+ опц.) out/phase_tie/<λ>nm/phi_tie.npy  — если есть, используем как фазу для центральной плоскости
Выход: out/phase_analytic/<λ>nm/{U_focus_analytic.npy, amplitude.png, phase.png, Pupil.npy, …}
"""

import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- ПАРАМЕТРЫ: ОБЯЗАТЕЛЬНО ВВЕДИТЕ СВОИ ----
from pathlib import Path
import json
CFG = Path("out")/"config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))
def na_for_lambda(l): 
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(l), cfg.get("na_default", 0.25)))
# и внутри цикла по λ: NA = na_for_lambda(lam_nm)


CLEAN_DIR = Path("out") / "clean"
CLEAN_ALIGNED = Path("out") / "clean_aligned"
if CLEAN_ALIGNED.is_dir():
    print("✅ Найден выровненный стек → использую out/clean_aligned")
    CLEAN_DIR = CLEAN_ALIGNED

TIE_PHASE_DIR  = Path("out") / "phase_tie"
OUT_DIR        = Path("out") / "phase_analytic"

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

def fresnel_refocus_to_focus(Uz: np.ndarray, dz_m: float, lam_m: float, dx_m: float) -> np.ndarray:
    """
    Аналитический перенос Uz(x,y) в фокус на -dz (параксиальный Fresnel).
    В частотной области: Û0 = Ûz * exp(+ i π λ dz (u^2+v^2)), u,v — cycles/m.
    """
    ny, nx = Uz.shape
    fx = np.fft.fftfreq(nx, d=dx_m)
    fy = np.fft.fftfreq(ny, d=dx_m)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    quad = np.exp(1j * np.pi * lam_m * dz_m * (FX**2 + FY**2))
    Uhat = np.fft.fft2(Uz)
    U0_hat = Uhat * quad
    U0 = np.fft.ifft2(U0_hat)
    return U0

def main():
    if DX_UM is None:
        raise SystemExit("Укажите DX_UM (микрон/пиксель) в начале файла.")
    ensure_dir(OUT_DIR)
    data = load_clean()
    for lam in sorted(data.keys()):
        d = data[lam]
        stack, z_um, idx = d["stack"], d["z_um"], d["idx_best"]
        lam_m = lam * 1e-9
        dx_m  = DX_UM * 1e-6
        NA    = na_for_lambda(lam)   # ← берём NA из out/config.json для текущей λ

        # Приоритет фазы: если есть TIE-фаза в центральной плоскости — используем её
        phi_path = TIE_PHASE_DIR / f"{lam}nm" / "phi_tie.npy"
        use_phi = phi_path.exists()
        if use_phi:
            phi0 = np.load(phi_path).astype(np.float64)

        # Скомбинируем несколько плоскостей: переносим каждую в фокус и усредняем
        U_focus_accum = None
        w_sum = 0.0
        z0 = z_um[idx]
        for k in range(stack.shape[0]):
            Ik = stack[k].astype(np.float64)
            amp = np.sqrt(Ik)
            if use_phi and k == idx:
                Uk = amp * np.exp(1j*phi0)   # есть фаза в центральной плоскости
            else:
                Uk = amp.astype(np.complex128)  # нулевая фаза (приближение)

            dz_m = (z_um[k] - z0) * 1e-6
            U0_k = fresnel_refocus_to_focus(Uk, dz_m=-dz_m, lam_m=lam_m, dx_m=dx_m)  # перенос в фокус

            if U_focus_accum is None:
                U_focus_accum = U0_k
            else:
                U_focus_accum += U0_k
            w_sum += 1.0

        U_focus = (U_focus_accum / max(w_sum, 1.0)).astype(np.complex64)

        out = OUT_DIR / f"{lam}nm"; ensure_dir(out)
        np.save(out / "U_focus_analytic.npy", U_focus)

        amp = np.abs(U_focus); amp /= (amp.max() + 1e-12)
        ph  = np.angle(U_focus)

        plt.figure(); plt.title(f"Amplitude (analytic) — {lam} nm"); plt.imshow(amp, cmap="gray"); plt.axis("off")
        plt.savefig(out / "amplitude.png", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure(); plt.title(f"Phase (analytic) — {lam} nm"); plt.imshow(ph, cmap="gray"); plt.axis("off")
        plt.savefig(out / "phase.png", dpi=150, bbox_inches="tight"); plt.close()

        # Pupil (жёсткая апертура)
        F = np.fft.fft2(U_focus)
        fx = np.fft.fftfreq(U_focus.shape[1], d=dx_m)
        fy = np.fft.fftfreq(U_focus.shape[0], d=dx_m)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        band = (FX**2 + FY**2 <= (NA/lam_m)**2)   # ← используем NA
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

        print(f"λ={lam} nm (analytic refocus, {'with' if use_phi else 'no'} phase, NA={NA:.3f}) → {out}")


if __name__ == "__main__":
    main()
