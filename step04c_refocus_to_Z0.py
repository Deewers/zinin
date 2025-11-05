# -*- coding: utf-8 -*-
"""
Шаг 4C (SIM). Генерация "рефокус" картинок для трёх методов (то, что показывает GUI).
Для каждого λ берём:
  - GS:       out/phase_gs/<λ>nm/U0.npy                  → I_ref = |U0|^2
  - TIE:      out/phase_tie/<λ>nm/U0.npy                 → I_ref = |U0|^2
  - Analytic: out/phase_analytic/<λ>nm/U_focus_analytic.npy → I_ref = |U_focus|^2
Сохраняем в:
  out/refocus/<method>/<λ>nm/I_ref.npy (+ .png)
"""

from pathlib import Path
import json, numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Конфиг (DX/NA не обязательны здесь, но читаем для сообщений) ----------
CFG = Path("out")/"config.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
DX_UM = float(cfg.get("dx_um", 3.3))
def na_for_lambda(lam_nm:int)->float:
    per = cfg.get("na_per_lambda", {}) or {}
    return float(per.get(str(lam_nm), cfg.get("na_default", 0.25)))

# ---------- Директории ----------
SIM_DIR   = Path("out")/"sim"
GS_DIR    = Path("out")/"phase_gs"
TIE_DIR   = Path("out")/"phase_tie"
AN_DIR    = Path("out")/"phase_analytic"
OUT_ROOT  = Path("out")/"refocus"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def normalize01(a: np.ndarray)->np.ndarray:
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (1.0, 99.5))
    if hi-lo < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if hi-lo < 1e-12: return np.zeros_like(a)
    a = (a-lo)/(hi-lo)
    return np.clip(a,0,1)

def available_wavelengths():
    lams = []
    if SIM_DIR.is_dir():
        for sub in sorted(SIM_DIR.iterdir()):
            if sub.is_dir() and sub.name.endswith("nm"):
                try: lams.append(int(sub.name[:-2]))
                except: pass
    return lams

def save_ref_image(out_dir: Path, I: np.ndarray, title: str):
    ensure_dir(out_dir)
    np.save(out_dir/"I_ref.npy", I.astype(np.float32))
    # PNG для быстрой визуализации
    plt.figure(); plt.title(title); plt.imshow(normalize01(I), cmap="gray"); plt.axis("off")
    plt.savefig(out_dir/"I_ref.png", dpi=150, bbox_inches="tight"); plt.close()

def main():
    lams = available_wavelengths()
    if not lams:
        raise SystemExit("Нет out/sim/*nm — сначала сгенерируйте симуляцию.")
    for lam in lams:
        NA = na_for_lambda(lam)

        # GS
        pU = GS_DIR/f"{lam}nm"/"U0.npy"
        if pU.exists():
            U = np.load(pU)
            I = np.abs(U)**2
            save_ref_image(OUT_ROOT/"phase_gs"/f"{lam}nm", I, f"Refocus GS — {lam} nm (NA={NA:.3f})")
            print(f"GS: λ={lam} nm → OK")
        else:
            print(f"GS: λ={lam} nm → нет {pU}")

        # TIE
        pU = TIE_DIR/f"{lam}nm"/"U0.npy"
        if pU.exists():
            U = np.load(pU)
            I = np.abs(U)**2
            save_ref_image(OUT_ROOT/"phase_tie"/f"{lam}nm", I, f"Refocus TIE — {lam} nm (NA={NA:.3f})")
            print(f"TIE: λ={lam} nm → OK")
        else:
            print(f"TIE: λ={lam} nm → нет {pU}")

        # Analytic
        pU = AN_DIR/f"{lam}nm"/"U_focus_analytic.npy"
        if pU.exists():
            U = np.load(pU)
            I = np.abs(U)**2
            save_ref_image(OUT_ROOT/"phase_analytic"/f"{lam}nm", I, f"Refocus Analytic — {lam} nm (NA={NA:.3f})")
            print(f"Analytic: λ={lam} nm → OK")
        else:
            print(f"Analytic: λ={lam} nm → нет {pU}")

    print("Готово. GUI вкладка Refocus теперь должна всё показывать.")

if __name__ == "__main__":
    main()
