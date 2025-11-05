# -*- coding: utf-8 -*-
"""
Шаг 2C. Синтетический Z‑стек из одной лучшей плоскости (±10 µm, шаг 1 µm).
U0 берём из лучшего фокуса: U0 = sqrt(I0) * exp(i*phi0), phi0 — из TIE, если есть.
Пропагируем коherent‑моделью (угловой спектр) с круговым pupil (NA из out/config.json).
Выход: out/sim/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Конфиг (DX и NA) ---
CFG = Path("out")/"config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))
def na_for_lambda(lam_nm:int)->float:
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(lam_nm), cfg.get("na_default", 0.25)))

# --- Директории ---
CLEAN_ALIGNED = Path("out")/"clean_aligned"
CLEAN = Path("out")/"clean"
TIE_DIR = Path("out")/"phase_tie"
SIM_DIR = Path("out")/"sim"

# --- Параметры симуляции ---
Z_MIN_UM = -10.0
Z_MAX_UM = +10.0
Z_STEP_UM= 1.0

# --- Угловой спектр (обобщённый на прямоугольные кадры) ---
class ASP:
    def __init__(self, lam_m:float, dx_m:float, ny:int, nx:int, na:float):
        self.lam = lam_m
        self.dx  = dx_m
        self.ny, self.nx = ny, nx
        fy = np.fft.fftfreq(ny, d=dx_m)
        fx = np.fft.fftfreq(nx, d=dx_m)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing='xy')
        self.F2 = self.FX**2 + self.FY**2
        self.k  = 2*np.pi/self.lam
        # круговой pupil по NA:
        self.band = (self.F2 <= (na/self.lam)**2).astype(np.float32)
        # предварительно считаем kz
        arg = np.maximum(1.0 - (self.lam**2)*self.F2, 0.0)
        self.kz = self.k*np.sqrt(arg)

    def propagate(self, U0:np.ndarray, z_m:float)->np.ndarray:
        H = np.exp(1j*self.kz*z_m) * self.band
        return np.fft.ifft2(np.fft.fft2(U0) * H)

def load_best_frame(lam_nm:int):
    # 1) предпочесть aligned, иначе clean
    base = CLEAN_ALIGNED if CLEAN_ALIGNED.is_dir() else CLEAN
    d = base / f"{lam_nm}nm"
    sp, zp, ip = d/"stack_clean.npy", d/"z_um.npy", d/"idx_best.npy"
    if not (sp.exists() and zp.exists()):
        return None
    stack = np.load(sp).astype(np.float64)
    z_um  = np.load(zp).astype(np.float64)
    idx_best = int(np.load(ip)[0]) if ip.exists() else int(np.argmax(stack.reshape(stack.shape[0], -1).sum(1)))
    I0 = stack[idx_best]
    return I0, z_um, idx_best

def tie_phase_if_any(lam_nm:int, shape):
    p = TIE_DIR/f"{lam_nm}nm"/"phi_tie.npy"
    if p.exists():
        phi = np.load(p).astype(np.float64)
        if phi.shape == shape:
            return phi
    return np.zeros(shape, dtype=np.float64)

def simulate_for_lambda(lam_nm:int):
    got = load_best_frame(lam_nm)
    if got is None:
        print(f"λ={lam_nm} nm: нет входных данных → пропуск")
        return
    I0, z_um_src, idx_best_src = got
    ny, nx = I0.shape
    lam_m = lam_nm*1e-9
    dx_m  = DX_UM*1e-6
    NA    = na_for_lambda(lam_nm)

    # U0 в фокусе
    amp0 = I0 / (I0.max() + 1e-12)
    amp0 = np.sqrt(amp0)
    phi0 = tie_phase_if_any(lam_nm, I0.shape)
    U0   = amp0 * np.exp(1j*phi0)

    # подготовка ASP
    asp = ASP(lam_m, dx_m, ny, nx, NA)

    # сетка z
    z_list_um = np.arange(Z_MIN_UM, Z_MAX_UM + 1e-9, Z_STEP_UM)
    z0_idx    = int(np.argmin(np.abs(z_list_um - 0.0)))  # индекс центрального слоя

    stack = np.empty((len(z_list_um), ny, nx), dtype=np.float32)
    for i, dz_um in enumerate(z_list_um):
        Uz = asp.propagate(U0, dz_um*1e-6)   # коherent
        Iz = np.abs(Uz)**2
        # мягкая нормализация кадра на 99.5 перцентиль (устойчиво к выбросам):
        vmax = np.percentile(Iz, 99.5)
        if vmax > 0: Iz = Iz / vmax
        stack[i] = Iz.astype(np.float32)

    out = SIM_DIR/f"{lam_nm}nm"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"stack_clean.npy", stack)
    np.save(out/"z_um.npy", z_list_um.astype(np.float32))
    np.save(out/"idx_best.npy", np.array([z0_idx], dtype=np.int32))

    # превью
    plt.figure(figsize=(6,3))
    plt.plot(z_list_um, stack.reshape(len(z_list_um), -1).mean(1))
    plt.axvline(0, ls="--", c="k")
    plt.xlabel("z (µm)"); plt.ylabel("⟨I⟩")
    plt.title(f"Simulated stack — λ={lam_nm} nm (NA={NA:.3f}, DX={DX_UM} µm/px)")
    plt.tight_layout()
    plt.savefig(out/"preview_intensity_vs_z.png", dpi=130)
    plt.close()

    print(f"λ={lam_nm} nm → out/sim/{lam_nm}nm  (frames={len(z_list_um)}, center idx={z0_idx})")

def main():
    base = CLEAN_ALIGNED if CLEAN_ALIGNED.is_dir() else CLEAN
    if not base.is_dir():
        raise SystemExit("Нет ни out/clean_aligned, ни out/clean — нечего симулировать.")
    # обрабатываем все доступные λ
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and sub.name.endswith("nm"):
            try:
                lam_nm = int(sub.name[:-2])
            except:
                continue
            simulate_for_lambda(lam_nm)

if __name__ == "__main__":
    main()
