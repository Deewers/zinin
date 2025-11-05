# -*- coding: utf-8 -*-
"""
GUI (SIM-ONLY): просмотр синтетических стеков и результатов шагов 3–4.
- Источник стеков: out/sim/<λ>nm/{stack_clean.npy, z_um.npy, idx_best.npy}
- Вкладки: Stacks, Focus, Phase/Field, Pupil, Validation, Refocus
- DX/NA берутся из out/config.json; Auto NA работает на сим-данных
- Устойчивая нормализация изображений (перцентили 1..99.5), без "белых/чёрных" кадров
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --- Matplotlib в Tk ---
import matplotlib as mpl
mpl.use("TkAgg")  # важно до импорта backend_tkagg
mpl.rcParams["font.size"] = 10.0
mpl.rcParams["axes.titlesize"] = 10.0

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- Конфиг ----------
CFG_PATH = Path("out") / "config.json"

def load_config() -> dict:
    if CFG_PATH.exists():
        try:
            return json.loads(CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # дефолт: DX=3.3, NA=0.25
    return {"dx_um": 3.3, "na_default": 0.25, "na_per_lambda": {}}

def save_config(cfg: dict):
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def na_for_lambda(cfg: dict, lam_nm: int) -> float:
    per = cfg.get("na_per_lambda", {}) or {}
    if str(lam_nm) in per:
        return float(per[str(lam_nm)])
    return float(cfg.get("na_default", 0.25))

# ---------- Утилиты нормализации/ROI ----------
def normalize01(img: np.ndarray) -> np.ndarray:
    """Устойчивая нормализация в [0,1] (перцентили 1..99.5), NaN/inf -> 0."""
    a = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    rng = float(a.max() - a.min())
    if not np.isfinite(rng) or rng < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(a, (1.0, 99.5))
    if (hi - lo) < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if (hi - lo) < 1e-12:
            return np.zeros_like(a, dtype=np.float32)
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0.0, 1.0)

def crop_center(img: np.ndarray, frac: float = 0.7) -> np.ndarray:
    assert 0 < frac <= 1.0
    H, W = img.shape
    h = int(round(H * frac))
    w = int(round(W * frac))
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return img[y0:y0 + h, x0:x0 + w]

# ---------- Данные проекта (SIM-ONLY) ----------
class ProjectData:
    def __init__(self, out_root: Path):
        self.set_root(out_root)

    def set_root(self, out_root: Path):
        self.root = Path(out_root)
        # работаем ТОЛЬКО с синтетикой
        self.dir_sim            = self.root / "sim"
        # результаты шагов 3–4
        self.dir_phase_gs       = self.root / "phase_gs"
        self.dir_phase_tie      = self.root / "phase_tie"
        self.dir_phase_analytic = self.root / "phase_analytic"
        self.dir_focus          = self.root / "focus"
        self.dir_validate       = self.root / "validate"
        self.dir_refocus        = self.root / "refocus"
        self._scan_wavelengths()

    def _scan_wavelengths(self):
        self.wavelengths = []
        base = self.dir_sim
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and sub.name.endswith("nm"):
                    try:
                        lam = int(sub.name[:-2])  # "595nm" -> 595
                        self.wavelengths.append(lam)
                    except:
                        pass

    def list_wavelengths(self) -> List[int]:
        return self.wavelengths

    # Стек берём ТОЛЬКО из out/sim/<λ>nm
    def load_clean_stack(self, lam_nm: int):
        d = self.dir_sim / f"{lam_nm}nm"
        sp = d / "stack_clean.npy"
        zp = d / "z_um.npy"
        ip = d / "idx_best.npy"
        if sp.exists() and zp.exists():
            stack = np.load(sp)
            z = np.load(zp).astype(float)
            idx = int(np.load(ip)[0]) if ip.exists() else int(np.argmin(np.abs(z - 0.0)))
            return stack, z, idx
        return None

    # Focus summary (опционально)
    def load_focus_summary(self) -> Optional[pd.DataFrame]:
        p = self.dir_focus / "summary.csv"
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    def consensus_z(self, lam_nm: int) -> Optional[float]:
        df = self.load_focus_summary()
        if df is None:
            return None
        row = df[df["lambda_nm"] == lam_nm]
        if not row.empty:
            return float(row["consensus_z_um"].iloc[0])
        return None

    # Комплексное поле
    def _load_field(self, base_dir: Path, lam_nm: int, fname: str) -> Optional[np.ndarray]:
        p = base_dir / f"{lam_nm}nm" / fname
        if p.exists():
            return np.load(p)
        return None

    def load_field(self, lam_nm: int, method: str) -> Optional[np.ndarray]:
        m = method.lower()
        if m == "gs":
            return self._load_field(self.dir_phase_gs, lam_nm, "U0.npy")
        if m == "tie":
            return self._load_field(self.dir_phase_tie, lam_nm, "U0.npy")
        if m == "analytic":
            return self._load_field(self.dir_phase_analytic, lam_nm, "U_focus_analytic.npy")
        return None

    # Pupil
    def load_pupil(self, lam_nm: int, method: str) -> Optional[np.ndarray]:
        base = {"gs": self.dir_phase_gs, "tie": self.dir_phase_tie, "analytic": self.dir_phase_analytic}.get(method.lower())
        if base is None:
            return None
        p = base / f"{lam_nm}nm" / "Pupil.npy"
        return np.load(p) if p.exists() else None

    # Validation
    def load_validation(self, lam_nm: int) -> Optional[pd.DataFrame]:
        p = self.dir_validate / f"{lam_nm}nm" / "validation.csv"
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    # Refocus
    def load_refocus(self, lam_nm: int, method: str) -> Optional[np.ndarray]:
        base = self.dir_refocus / method.lower() / f"{lam_nm}nm"
        p = base / "I_ref.npy"
        if p.exists():
            return np.load(p)
        png = base / "I_ref.png"
        if png.exists():
            img = mpimg.imread(png)
            if img.ndim == 3:
                img = img[..., 0]
            return img
        return None

# ---------- Рисовальная панель ----------
class FigurePanel:
    def __init__(self, parent, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def show_image(self, img: np.ndarray, title: str = "", cmap="gray", vmin=None, vmax=None, do_norm=True):
        self.ax.clear()
        self.ax.set_title(title)
        if do_norm and (vmin is None or vmax is None):
            arr = normalize01(img)
            self.ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
        else:
            arr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            self.ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def show_plot(self, x: np.ndarray, ys: Dict[str, np.ndarray], vline: Optional[float] = None, xlabel="x", ylabel="y", title=""):
        self.ax.clear()
        self.ax.set_title(title)
        for name, y in ys.items():
            self.ax.plot(x, y, label=name)
        if vline is not None:
            self.ax.axvline(vline, ls="--", label=f"Z0={vline:.2f} μm")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(alpha=0.3)
        self.ax.legend()
        self.canvas.draw_idle()

# ---------- Метрики резкости ----------
def variance_of_laplacian(img: np.ndarray) -> float:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    H, W = img.shape
    F_img = np.fft.fft2(img)
    F_k   = np.fft.fft2(k, s=(H, W))
    conv = np.fft.ifft2(F_img * F_k).real
    return float(conv.var())

def gradient_energy(img: np.ndarray) -> float:
    gx = (np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)) / 2.0
    gy = (np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)) / 2.0
    return float((gx * gx + gy * gy).mean())

def highfreq_fft_energy(img: np.ndarray, frac: float = 0.25) -> float:
    F = np.fft.fftshift(np.fft.fft2(img))
    mag2 = (np.abs(F) ** 2)
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    mask = (r > frac * rmax)
    return float(mag2[mask].mean())

def sharpness_curves(stack: np.ndarray) -> Dict[str, np.ndarray]:
    s1, s2, s3 = [], [], []
    for k in range(stack.shape[0]):
        img = normalize01(stack[k])
        s1.append(variance_of_laplacian(img))
        s2.append(gradient_energy(img))
        s3.append(highfreq_fft_energy(img))
    return {"Laplacian": np.array(s1), "Gradient": np.array(s2), "FFT High‑freq": np.array(s3)}

# ---------- Модель (угловой спектр) и сверки для Auto NA ----------
class AngularSpectrum2D:
    def __init__(self, lam_m: float, dx_m: float, ny: int, nx: int, na: float):
        self.lam = lam_m
        self.dx  = dx_m
        self.ny, self.nx = ny, nx
        fy = np.fft.fftfreq(ny, d=dx_m)
        fx = np.fft.fftfreq(nx, d=dx_m)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing="xy")
        self.F2 = self.FX**2 + self.FY**2
        self.k  = 2 * np.pi / self.lam
        self.band = (self.F2 <= (na / self.lam)**2).astype(np.float32)
        arg = np.maximum(1.0 - (self.lam**2) * self.F2, 0.0)
        self.kz = self.k * np.sqrt(arg)

    def H(self, z_m: float):
        return np.exp(1j * self.kz * z_m) * self.band

def synthesize_psf_from_na(ny: int, nx: int, dz_m: float, lam_m: float, dx_m: float, na: float) -> np.ndarray:
    asp = AngularSpectrum2D(lam_m, dx_m, ny, nx, na)
    Uz = np.fft.ifft2(asp.band * asp.H(dz_m))
    I  = np.abs(Uz) ** 2
    vmax = np.percentile(I, 99.5)
    if vmax > 0:
        I = I / vmax
    return I.astype(np.float32)

def fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b)).real

def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    da = np.sqrt((a * a).mean()) + 1e-12
    db = np.sqrt((b * b).mean()) + 1e-12
    return float((a * b).mean() / (da * db))

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SIM Viewer — Pupil/PSF/Refocus")
        self.geometry("1320x840")

        # модель и конфиг
        self.cfg = load_config()
        self.project = ProjectData(Path("./out"))

        # Верхняя панель
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Папка out:").pack(side=tk.LEFT)
        self.out_path_var = tk.StringVar(value=str(self.project.root))
        self.out_entry = ttk.Entry(top, textvariable=self.out_path_var, width=60); self.out_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Обзор…", command=self.browse_out).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Обновить", command=self.refresh_project).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="λ (nm):").pack(side=tk.LEFT, padx=(16, 3))
        self.lam_var = tk.StringVar(value="")
        self.lam_combo = ttk.Combobox(top, textvariable=self.lam_var, width=8, state="readonly")
        self.lam_combo.pack(side=tk.LEFT)
        self.lam_combo.bind("<<ComboboxSelected>>", lambda e: self.update_all_tabs())

        # DX / NA контроль
        ttk.Label(top, text="DX (µm/px):").pack(side=tk.LEFT, padx=(16, 3))
        self.dx_var = tk.StringVar(value=str(self.cfg.get("dx_um", 3.3)))
        self.dx_entry = ttk.Entry(top, textvariable=self.dx_var, width=6); self.dx_entry.pack(side=tk.LEFT)

        ttk.Label(top, text="NA:").pack(side=tk.LEFT, padx=(12, 3))
        self.na_var = tk.StringVar(value=str(self.cfg.get("na_default", 0.25)))
        self.na_entry = ttk.Entry(top, textvariable=self.na_var, width=6); self.na_entry.pack(side=tk.LEFT)

        ttk.Button(top, text="Auto NA (для λ)", command=self.auto_na_for_current).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Сохранить", command=self.save_dx_na).pack(side=tk.LEFT, padx=4)

        # Notebook
        self.nb = ttk.Notebook(self); self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab: Stacks
        self.tab_stacks = ttk.Frame(self.nb); self.nb.add(self.tab_stacks, text="Stacks (SIM)")
        self.stack_ctrl = ttk.Frame(self.tab_stacks); self.stack_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        self.stack_fig_frame = ttk.Frame(self.tab_stacks); self.stack_fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stack_panel = FigurePanel(self.stack_fig_frame, width=6, height=5)

        self.show_raw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.stack_ctrl, text="Raw (отключено в синтетике)",
                        variable=self.show_raw_var, state="disabled").pack(anchor=tk.W, pady=4)

        ttk.Label(self.stack_ctrl, text="z-index:").pack(anchor=tk.W)
        self._slider_guard = False
        self.z_slider = ttk.Scale(self.stack_ctrl, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_moved)
        self.z_slider.pack(fill=tk.X, pady=4)
        self.z_label = ttk.Label(self.stack_ctrl, text="z = —"); self.z_label.pack(anchor=tk.W)

        # Tab: Focus
        self.tab_focus = ttk.Frame(self.nb); self.nb.add(self.tab_focus, text="Focus")
        self.focus_fig_frame = ttk.Frame(self.tab_focus); self.focus_fig_frame.pack(fill=tk.BOTH, expand=True)
        self.focus_panel = FigurePanel(self.focus_fig_frame, width=7, height=5)

        # Tab: Phase/Field
        self.tab_phase = ttk.Frame(self.nb); self.nb.add(self.tab_phase, text="Phase/Field")
        self.phase_figs = ttk.Panedwindow(self.tab_phase, orient=tk.HORIZONTAL); self.phase_figs.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        left = ttk.Frame(self.phase_figs); right = ttk.Frame(self.phase_figs)
        self.phase_figs.add(left, weight=1); self.phase_figs.add(right, weight=1)
        self.method_var = tk.StringVar(value="GS")
        mrow = ttk.Frame(self.tab_phase); mrow.pack(side=tk.TOP, fill=tk.X, padx=8)
        ttk.Label(mrow, text="Метод:").pack(side=tk.LEFT)
        mcb = ttk.Combobox(mrow, textvariable=self.method_var, values=["GS", "TIE", "Analytic"], width=10, state="readonly")
        mcb.pack(side=tk.LEFT, padx=6)
        self.method_var.trace_add("write", lambda *_: self.update_phase_tabs())
        self.amp_panel = FigurePanel(left, 5, 4); self.ph_panel = FigurePanel(right, 5, 4)

        # Tab: Pupil
        self.tab_pupil = ttk.Frame(self.nb); self.nb.add(self.tab_pupil, text="Pupil")
        self.pupil_figs = ttk.Panedwindow(self.tab_pupil, orient=tk.HORIZONTAL); self.pupil_figs.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        leftp = ttk.Frame(self.pupil_figs); rightp = ttk.Frame(self.pupil_figs)
        self.pupil_figs.add(leftp, weight=1); self.pupil_figs.add(rightp, weight=1)
        self.pupil_amp_panel = FigurePanel(leftp, 5, 4); self.pupil_ph_panel = FigurePanel(rightp, 5, 4)

        # Tab: Validation
        self.tab_val = ttk.Frame(self.nb); self.nb.add(self.tab_val, text="Validation")
        topv = ttk.Frame(self.tab_val); topv.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(topv, text="Открыть папку валидации", command=self.open_validation_folder).pack(side=tk.LEFT)
        self.val_fig_frame = ttk.Frame(self.tab_val); self.val_fig_frame.pack(fill=tk.BOTH, expand=True)
        self.val_panel = FigurePanel(self.val_fig_frame, 7, 5)

        # Tab: Refocus
        self.tab_ref = ttk.Frame(self.nb); self.nb.add(self.tab_ref, text="Refocus")
        ctrl = ttk.Frame(self.tab_ref); ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(ctrl, text="Метод:").pack(side=tk.LEFT)
        self.ref_method_var = tk.StringVar(value="phase_gs")
        rcb = ttk.Combobox(ctrl, textvariable=self.ref_method_var, values=["phase_gs", "phase_tie", "phase_analytic"], state="readonly", width=14)
        rcb.pack(side=tk.LEFT, padx=4)
        self.ref_method_var.trace_add("write", lambda *_: self.update_refocus_tab())
        self.ref_fig_frame = ttk.Frame(self.tab_ref); self.ref_fig_frame.pack(fill=tk.BOTH, expand=True)
        self.ref_panel = FigurePanel(self.ref_fig_frame, 6, 5)

        # Статус
        self.status = tk.StringVar(value="Готово.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=3)

        # init
        self.refresh_project()

    # ---- конфиг/общие ----
    def browse_out(self):
        d = filedialog.askdirectory(title="Выберите папку out")
        if d:
            self.out_path_var.set(d)
            self.refresh_project()

    def refresh_project(self):
        try:
            root = Path(self.out_path_var.get())
            self.project.set_root(root)
            lams = self.project.list_wavelengths()
            self.lam_combo["values"] = [str(x) for x in lams]
            if lams:
                self.lam_var.set(str(lams[0]))
            else:
                self.lam_var.set("")
            # подтягиваем dx/na_default из конфига
            self.cfg = load_config()
            self.dx_var.set(str(self.cfg.get("dx_um", 3.3)))
            self.na_var.set(str(self.cfg.get("na_default", 0.25)))
            self.update_all_tabs()
            self.status.set(f"Загружено out: {root}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def current_lambda(self) -> Optional[int]:
        try:
            return int(self.lam_var.get())
        except:
            return None

    def on_slider_moved(self, _evt=None):
        # избегаем рекурсий
        if self._slider_guard:
            return
        # безопасное обновление после завершения события скролла
        self.after_idle(self.update_stack_view)

    # ---- Stacks (SIM) ----
    def update_stack_view(self):
        lam = self.current_lambda()
        if lam is None:
            self.stack_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            return

        clean = self.project.load_clean_stack(lam)
        if clean is None:
            self.status.set(f"Нет out/sim/{lam}nm — запустите step02c_simulate_stack_from_best.py")
            self.stack_panel.show_image(np.zeros((10, 10)), title="Нет данных (sim)")
            return

        stack, z, idx_best = clean
        n_frames = int(stack.shape[0])
        desired_to = max(0, n_frames - 1)

        # обновляем диапазон слайдера только при необходимости
        try:
            current_to = int(float(self.z_slider.cget("to")))
        except Exception:
            current_to = -1
        if current_to != desired_to:
            self._slider_guard = True
            self.z_slider.configure(from_=0, to=desired_to)
            self.z_slider.set(int(idx_best))  # центр — 0 мкм
            self._slider_guard = False

        try:
            k = int(round(float(self.z_slider.get())))
        except Exception:
            k = int(idx_best)
        k = max(0, min(desired_to, k))

        z_val = float(z[k]) if (z is not None and len(z) == n_frames) else float(k)
        self.z_label.config(text=f"z = {z_val:.2f} μm (index {k})")

        self.stack_panel.show_image(stack[k], title=f"Sim — λ={lam} nm", cmap="gray")

    # ---- Focus ----
    def update_focus_tab(self):
        lam = self.current_lambda()
        if lam is None:
            self.focus_panel.show_plot(np.array([0]), {"empty": np.array([0])}, title="Нет λ")
            return
        clean = self.project.load_clean_stack(lam)
        if clean is None:
            self.focus_panel.show_plot(np.array([0]), {"empty": np.array([0])}, title=f"Нет out/sim/{lam}nm")
            return
        stack, z, _ = clean
        curves = sharpness_curves(stack)
        z0 = self.project.consensus_z(lam)
        self.focus_panel.show_plot(z, curves, vline=z0, xlabel="z (μm)", ylabel="sharpness", title=f"Sharpness — λ={lam} nm")

    # ---- Phase/Field + Pupil ----
    def update_phase_tabs(self):
        lam = self.current_lambda()
        if lam is None:
            self.amp_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            self.ph_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            self.pupil_amp_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            self.pupil_ph_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            return

        method = self.method_var.get()
        U = self.project.load_field(lam, method)
        if U is None:
            self.amp_panel.show_image(np.zeros((10, 10)), title=f"Нет поля ({method})")
            self.ph_panel.show_image(np.zeros((10, 10)), title=f"Нет поля ({method})")
            self.pupil_amp_panel.show_image(np.zeros((10, 10)), title=f"Нет pupil ({method})")
            self.pupil_ph_panel.show_image(np.zeros((10, 10)), title=f"Нет pupil ({method})")
            return

        amp = np.abs(U).astype(np.float32)
        self.amp_panel.show_image(amp, title=f"Amplitude — λ={lam} nm ({method})", cmap="gray")

        ph = np.angle(U).astype(np.float32)
        self.ph_panel.show_image(ph, title=f"Phase — λ={lam} nm ({method})",
                                 cmap="twilight", vmin=-np.pi, vmax=np.pi, do_norm=False)

        P = self.project.load_pupil(lam, method)
        if P is not None:
            Pamp = np.abs(P).astype(np.float32)
            self.pupil_amp_panel.show_image(np.fft.fftshift(Pamp), title=f"Pupil amplitude — {method}", cmap="gray")

            Pph = np.angle(P).astype(np.float32)
            self.pupil_ph_panel.show_image(np.fft.fftshift(Pph), title=f"Pupil phase — {method}",
                                           cmap="twilight", vmin=-np.pi, vmax=np.pi, do_norm=False)
        else:
            self.pupil_amp_panel.show_image(np.zeros_like(amp), title=f"Нет pupil ({method})")
            self.pupil_ph_panel.show_image(np.zeros_like(amp), title=f"Нет pupil ({method})")

    # ---- Validation ----
    def update_validation_tab(self):
        lam = self.current_lambda()
        if lam is None:
            self.val_panel.show_plot(np.array([0.0]), {"—": np.array([0.0])}, title="Нет λ")
            return
        df = self.project.load_validation(lam)
        if df is None or ("dz_um" not in df.columns):
            self.val_panel.show_plot(np.array([0.0]), {"—": np.array([0.0])}, title="Нет validation.csv")
            return
        x = df["dz_um"].values
        ys = {}
        if "pearson_r" in df.columns:
            ys["Pearson r"] = df["pearson_r"].values
        if "mse" in df.columns:
            m = df["mse"].values
            if np.max(m) > 0:
                m = m / np.max(m)
            ys["MSE (norm)"] = m
        na_used = na_for_lambda(self.cfg, lam)
        self.val_panel.show_plot(x, ys, xlabel="Δz (μm)", ylabel="metric", title=f"Validation — λ={lam} nm (NA={na_used:.3f})")

    def open_validation_folder(self):
        lam = self.current_lambda()
        if lam is None:
            return
        folder = self.project.dir_validate / f"{lam}nm"
        if folder.exists():
            try:
                os.startfile(str(folder))
            except Exception:
                messagebox.showinfo("Папка", str(folder))
        else:
            messagebox.showinfo("Нет папки", f"Не найдено: {folder}")

    # ---- Refocus ----
    def update_refocus_tab(self):
        lam = self.current_lambda()
        if lam is None:
            self.ref_panel.show_image(np.zeros((10, 10)), title="Нет λ")
            return
        method = self.ref_method_var.get()
        I = self.project.load_refocus(lam, method)
        if I is None:
            self.ref_panel.show_image(np.zeros((10, 10)), title=f"Нет refocus ({method})")
            return
        na_used = na_for_lambda(self.cfg, lam)
        self.ref_panel.show_image(I, title=f"Refocused — λ={lam} nm ({method}), NA={na_used:.3f}", cmap="gray")

    # ---- DX/NA: авто-оценка и сохранение ----
    def auto_na_for_current(self):
        lam = self.current_lambda()
        if lam is None:
            messagebox.showwarning("λ не выбрана", "Сначала выберите длину волны.")
            return
        clean = self.project.load_clean_stack(lam)
        if clean is None:
            messagebox.showwarning("Нет данных", f"Нет out/sim/{lam}nm")
            return
        stack, z_um, idx_best = clean
        try:
            dx_um = float(self.dx_var.get())
        except Exception:
            messagebox.showerror("DX", "DX (µm/px) задан некорректно.")
            return
        lam_m = lam * 1e-9
        dx_m = dx_um * 1e-6

        # кандидатный диапазон NA
        na_grid = np.arange(0.05, 0.501, 0.01)
        K = min(5, idx_best, stack.shape[0] - 1 - idx_best)
        if K < 1:
            messagebox.showwarning("Мало плоскостей", "Нужно минимум по одной плоскости по обе стороны от лучшего фокуса.")
            return
        z_idxs = list(range(idx_best - K, idx_best + K + 1))
        R0 = normalize01(stack[idx_best].astype(np.float64))

        best_na, best_score = None, -1.0
        for na in na_grid:
            rs = []
            for k in z_idxs:
                dz_m = (float(z_um[k]) - float(z_um[idx_best])) * 1e-6
                psf = synthesize_psf_from_na(R0.shape[0], R0.shape[1], dz_m, lam_m, dx_m, na)
                pred = fft_convolve2d(R0, psf)
                pred = normalize01(pred)
                meas = normalize01(stack[k])
                # сравниваем в центральном ROI
                rs.append(pearsonr(crop_center(pred, 0.7), crop_center(meas, 0.7)))
            score = float(np.mean(rs))
            if score > best_score:
                best_score, best_na = score, float(na)

        if best_na is None:
            messagebox.showerror("NA", "Не удалось оценить NA.")
            return

        self.na_var.set(f"{best_na:.4f}")
        self.status.set(f"Авто NA для λ={lam} nm → {best_na:.4f} (сред. r={best_score:.3f})")
        messagebox.showinfo("Auto NA", f"Оценка NA для λ={lam} nm: {best_na:.4f}\n"
                                       f"(ср. корреляция={best_score:.3f}).\nНажмите «Сохранить», чтобы записать в config.")

    def save_dx_na(self):
        try:
            dx_um = float(self.dx_var.get())
        except Exception:
            messagebox.showerror("DX", "DX (µm/px) задан некорректно.")
            return
        try:
            na_val = float(self.na_var.get())
        except Exception:
            messagebox.showerror("NA", "NA задана некорректно.")
            return

        cfg = load_config()
        cfg["dx_um"] = dx_um
        cfg["na_default"] = na_val
        lam = self.current_lambda()
        if lam is not None:
            per = cfg.get("na_per_lambda", {}) or {}
            per[str(lam)] = na_val
            cfg["na_per_lambda"] = per

        save_config(cfg)
        self.cfg = cfg  # обновим у себя
        self.status.set(f"Сохранено: DX={dx_um} µm/px; NA_default={na_val}; NA[{lam}nm]={na_val}")
        self.update_validation_tab()
        self.update_refocus_tab()

    # ---- Массовый апдейт ----
    def update_all_tabs(self):
        self.update_stack_view()
        self.update_focus_tab()
        self.update_phase_tabs()
        self.update_validation_tab()
        self.update_refocus_tab()

if __name__ == "__main__":
    app = App()
    app.mainloop()
