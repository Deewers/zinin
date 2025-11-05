# -*- coding: utf-8 -*-
"""
Шаг 2B. Регистрация Z‑стека (выравнивание по сдвигу) для каждой длины волны.
- Референс: кадр в лучшем фокусе (idx_best).
- Оценка сдвига: фазовая корреляция на центральном ROI с Тьюки-окном.
- Применение: субпиксельный Fourier‑сдвиг (без смазывания).
Выход: out/clean_aligned/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy, shifts_xy.npy, drift.png, drift.csv}
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Конфиг DX (для оценки угла) --------
CFG = Path("out")/"config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))  # µm/px

# -------- Параметры регистрации --------
ROI_FRAC = 0.70      # центральное окно (доля по каждой оси), 0<...<=1
TUKEY_ALPHA = 0.25   # параметр окна Тьюки (0..1)
USE_SUBPIX = True    # субпиксельная оценка пика
EPS = 1e-12

CLEAN_DIR = Path("out") / "clean"
OUT_DIR   = Path("out") / "clean_aligned"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ---------- окна/ROI ----------
def tukey_1d(n: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    x = np.linspace(0, 1, n, endpoint=False)
    w = np.ones_like(x)
    edge = alpha/2
    m1 = (x < edge)
    m2 = (x >= (1 - edge))
    w[m1] = 0.5*(1 + np.cos(np.pi*(2*x[m1]/alpha - 1)))
    w[m2] = 0.5*(1 + np.cos(np.pi*(2*x[m2]/alpha - 2/alpha + 1)))
    return w

def tukey_2d(h: int, w: int, alpha: float) -> np.ndarray:
    return np.outer(tukey_1d(h, alpha), tukey_1d(w, alpha))

def crop_roi(img: np.ndarray, frac: float):
    assert 0 < frac <= 1.0
    h, w = img.shape
    rh = int(round(h*frac))
    rw = int(round(w*frac))
    y0 = (h - rh)//2
    x0 = (w - rw)//2
    return img[y0:y0+rh, x0:x0+rw], (y0, x0)

# ---------- фазовая корреляция ----------
def peak_subpixel_1d(fm1, f0, fp1):
    # параболическая интерполяция пика, возвращает сдвиг в (-1..+1)
    denom = (fm1 - 2*f0 + fp1)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom

def phase_correlation_shift(ref: np.ndarray, img: np.ndarray, roi_frac=ROI_FRAC, alpha=TUKEY_ALPHA, subpix=USE_SUBPIX):
    # подготавливаем ROI и окна
    ref_roi, (ry0, rx0) = crop_roi(ref, roi_frac)
    img_roi, (iy0, ix0) = crop_roi(img, roi_frac)
    h, w = ref_roi.shape
    W = tukey_2d(h, w, alpha)

    a = (ref_roi - ref_roi.mean()) * W
    b = (img_roi - img_roi.mean()) * W

    FA = np.fft.fft2(a)
    FB = np.fft.fft2(b)
    R = FA * np.conj(FB)
    R /= (np.abs(R) + EPS)
    corr = np.fft.ifft2(R)
    corr_abs = np.abs(corr)

    # пик (целочисленный)
    p = np.unravel_index(int(np.argmax(corr_abs)), corr_abs.shape)
    py, px = int(p[0]), int(p[1])
    if py > h//2: dy = py - h
    else:         dy = py
    if px > w//2: dx = px - w
    else:         dx = px

    # субпиксельная поправка (по каждой оси отдельно)
    if subpix:
        # с оборачиванием индексов
        ym1, y0, yp1 = (py-1) % h, py % h, (py+1) % h
        xm1, x0, xp1 = (px-1) % w, px % w, (px+1) % w
        dy += peak_subpixel_1d(corr_abs[ym1, px], corr_abs[y0, px], corr_abs[yp1, px])
        dx += peak_subpixel_1d(corr_abs[py, xm1], corr_abs[py, x0], corr_abs[py, xp1])

    # это сдвиг ROI изображения img относительно ROI ref;
    # чтобы привести img к ref, нужно сдвинуть img на (-dy, -dx) в координатах ROI.
    # Так как ROI по центру, этот сдвиг справедлив для целого кадра:
    return float(dy), float(dx)

# ---------- Fourier‑сдвиг изображения на (dy, dx) ----------
def fourier_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    h, w = img.shape
    ky = np.fft.fftfreq(h).reshape(-1, 1)  # [-0.5..0.5)
    kx = np.fft.fftfreq(w).reshape(1, -1)
    phase = np.exp(-2j*np.pi*(ky*dy + kx*dx))
    F = np.fft.fft2(img)
    shifted = np.fft.ifft2(F * phase)
    return np.real(shifted).astype(img.dtype)

def process_lambda(lam_nm: int, stack: np.ndarray, z_um: np.ndarray, idx_best: int, out_dir: Path):
    ensure_dir(out_dir)
    ref = stack[idx_best].astype(np.float64)

    shifts = []
    aligned = np.empty_like(stack, dtype=np.float32)

    for k in range(stack.shape[0]):
        img = stack[k].astype(np.float64)
        dy, dx = phase_correlation_shift(ref, img, ROI_FRAC, TUKEY_ALPHA, USE_SUBPIX)
        # применяем сдвиг к ИСХОДНОМУ (без окон/ROI) изображению
        aligned[k] = fourier_shift(stack[k].astype(np.float64), -dy, -dx).astype(np.float32)
        shifts.append((float(dy), float(dx)))

    shifts = np.asarray(shifts, dtype=np.float64)  # [Z, 2] — dy, dx (в пикселях, оцененные на ROI)
    np.save(out_dir/"stack_clean.npy", aligned.astype(np.float32))
    np.save(out_dir/"z_um.npy", z_um.astype(np.float32))
    np.save(out_dir/"idx_best.npy", np.array([idx_best], dtype=np.int32))
    np.save(out_dir/"shifts_xy.npy", shifts.astype(np.float32))

    # график дрейфа и оценка угла
    dz = z_um - float(z_um[idx_best])
    dy, dx = shifts[:,0], shifts[:,1]
    # линейная аппроксимация dy/dz, dx/dz
    A = np.vstack([dz, np.ones_like(dz)]).T
    sy, _ = np.linalg.lstsq(A, dy, rcond=None)[0]  # px per µm
    sx, _ = np.linalg.lstsq(A, dx, rcond=None)[0]
    # модуль сдвига на 1 µm по z, в мкм/µм:
    drift_um_per_um = float(np.hypot(sx, sy) * DX_UM)
    tilt_deg = float(np.degrees(np.arctan(drift_um_per_um)))  # прибл. угол наклона образца/оптики

    # сохраняем CSV
    import csv
    with open(out_dir/"drift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z_um","dy_px","dx_px"])
        for zi, yi, xi in zip(z_um, dy, dx):
            w.writerow([float(zi), float(yi), float(xi)])

    # рисуем
    plt.figure(figsize=(6,4))
    plt.plot(dz, dy, label="dy (px)")
    plt.plot(dz, dx, label="dx (px)")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Δz (µm)"); plt.ylabel("shift (px)")
    plt.title(f"{lam_nm} nm — drift vs z   |   tilt≈{tilt_deg:.2f}°")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"drift.png", dpi=150)
    plt.close()

    print(f"λ={lam_nm} nm: aligned → {out_dir}  |  tilt≈{tilt_deg:.2f}°  (|d(x,y)/dz|·DX={drift_um_per_um:.3f} µm/µm)")

def main():
    if not CLEAN_DIR.is_dir():
        raise SystemExit(f"Нет {CLEAN_DIR} — сначала выполните шаги 1–2.")
    ensure_dir(OUT_DIR)

    for sub in sorted(CLEAN_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("nm"):
            continue
        lam_nm = int(sub.name[:-2])
        stack = np.load(sub/"stack_clean.npy")      # [Z,H,W]
        z_um  = np.load(sub/"z_um.npy").astype(np.float64)
        idxp  = sub/"idx_best.npy"
        idx_best = int(np.load(idxp)[0]) if idxp.exists() else int(np.argmax(stack.reshape(stack.shape[0], -1).sum(1)))

        out_dir = OUT_DIR / f"{lam_nm}nm"
        process_lambda(lam_nm, stack, z_um, idx_best, out_dir)

if __name__ == "__main__":
    main()
