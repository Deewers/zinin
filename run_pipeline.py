# -*- coding: utf-8 -*-
"""
Оркестратор: запускает весь пайплайн по шагам.

Шаги:
  01  step01_ingest_build_stacks.py
  02  step02_preprocess.py                 ← добавлен
  02b step02b_register_stacks.py
  02c step02c_simulate_stack_from_best.py
  03a step03a_phase_gs.py
  03b step03b_phase_tie.py
  03c step03c_phase_analytic.py
  04a step04a_focus_summary.py
  04b step04b_validate_psf.py
  04c step04c_refocus_to_Z0.py

Особенности:
- Логи в out/logs/<step>.log
- Если есть out/sim и включён USE_SIM_FOR_STEP3=True, шаги 3A/3B/3C берут стек из out/sim.
- Иначе — предпочитается out/clean_aligned, затем out/clean.
"""

import sys, os, json, subprocess, traceback
from datetime import datetime
from pathlib import Path
import importlib.util

# --------------------------- НАСТРОЙКИ ---------------------------

# Какие шаги выполнять
RUN = {
    "01_ingest":        True,
    "02_preprocess":    True,   # ← новый шаг
    "02b_align":        True,
    "02c_simulate":     True,   # хотим симуляцию ±10 мкм
    "03a_gs":           True,
    "03b_tie":          True,
    "03c_analytic":     True,
    "04a_focus":        True,
    "04b_validate":     True,
    "04c_refocus":      True,
}

STOP_ON_ERROR = True      # останавливать ли конвейер при первом падении
USE_SIM_FOR_STEP3 = True  # для шагов 3A/3B/3C использовать out/sim, если существует
PREFER_ALIGNED = True     # если нет sim, предпочесть out/clean_aligned вместо out/clean

# Ваш DX и NA по умолчанию (если нет out/config.json)
DEFAULT_DX_UM = 3.3
DEFAULT_NA    = 0.25

# Имена файлов шагов (ожидаются рядом с этим скриптом)
STEP_SCRIPTS = {
    "01_ingest":    "step01_ingest_build_stacks.py",
    "02_preprocess":"step02_preprocess.py",          # ← добавлен
    "02b_align":    "step02b_register_stacks.py",
    "02c_simulate": "step02c_simulate_stack_from_best.py",
    "03a_gs":       "step03a_phase_gs.py",
    "03b_tie":      "step03b_phase_tie.py",
    "03c_analytic": "step03c_phase_analytic.py",
    "04a_focus":    "step04a_focus_summary.py",
    "04b_validate": "step04b_validate_psf.py",
    "04c_refocus":  "step04c_refocus_to_Z0.py",
}

# ----------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUT  = HERE / "out"
LOGS = OUT / "logs"
CFG  = OUT / "config.json"

def ensure_out_and_config():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if not CFG.exists():
        cfg = {"dx_um": DEFAULT_DX_UM, "na_default": DEFAULT_NA, "na_per_lambda": {}}
        CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[init] Создан {CFG} (dx_um={DEFAULT_DX_UM}, na_default={DEFAULT_NA})")
    else:
        try:
            _ = json.loads(CFG.read_text(encoding="utf-8"))
        except Exception:
            # если файл битый — заменим дефолтом
            cfg = {"dx_um": DEFAULT_DX_UM, "na_default": DEFAULT_NA, "na_per_lambda": {}}
            CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[init] Повреждённый {CFG} — перезаписан дефолтом.")

def log_path(step_key: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS / f"{ts}_{step_key}.log"

def run_subprocess(step_key: str, script_path: Path, args=None, extra_env=None) -> int:
    """Запускает шаг отдельным процессом, пишет stdout+stderr в лог."""
    args = args or []
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("MPLBACKEND", "Agg")  # без окон

    log_file = log_path(step_key)
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print(f"[run] {step_key}: {cmd}\n      лог → {log_file}")
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"CMD: {' '.join(cmd)}\nCWD: {HERE}\n\n"); f.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line); f.write(line)
        proc.wait()
        rc = proc.returncode
        f.write(f"\n[exit] returncode={rc}\n")
        print(f"[exit] {step_key}: returncode={rc}")
        return rc

def import_and_run_with_clean_dir(step_key: str, script_path: Path, clean_dir: Path) -> int:
    """Импортирует модуль, подменяет module.CLEAN_DIR, вызывает module.main()."""
    log_file = log_path(step_key)
    print(f"[run] {step_key}: import {script_path.name} (CLEAN_DIR={clean_dir})\n      лог → {log_file}")
    with log_file.open("w", encoding="utf-8") as f:
        try:
            spec = importlib.util.spec_from_file_location(f"_{step_key}", str(script_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

            if hasattr(mod, "CLEAN_DIR"):
                setattr(mod, "CLEAN_DIR", clean_dir)
                f.write(f"Patched CLEAN_DIR = {clean_dir}\n")
            else:
                f.write("WARNING: module has no CLEAN_DIR; nothing to patch.\n")

            if hasattr(mod, "main"):
                old_stdout = sys.stdout
                class Tee:
                    def __init__(self, a, b): self.a, self.b = a, b
                    def write(self, s): self.a.write(s); self.b.write(s)
                    def flush(self): self.a.flush(); self.b.flush()
                sys.stdout = Tee(sys.stdout, f)
                try:
                    mod.main()
                    rc = 0
                finally:
                    sys.stdout = old_stdout
                f.write("\n[exit] returncode=0\n")
                print(f"[exit] {step_key}: returncode=0")
                return 0
            else:
                f.write("ERROR: module has no main()\n")
                print(f"[exit] {step_key}: module has no main()")
                return 1
        except Exception:
            f.write(f"\nEXCEPTION:\n{traceback.format_exc()}\n")
            print(f"[error] {step_key}: exception, см. лог {log_file}")
            return 1

def choose_clean_dir_for_step3() -> Path:
    """Выбор источника стэка для шагов 3A/3B/3C."""
    sim = OUT/"sim"
    aligned = OUT/"clean_aligned"
    clean = OUT/"clean"
    if USE_SIM_FOR_STEP3 and sim.is_dir():
        return sim
    if PREFER_ALIGNED and aligned.is_dir():
        return aligned
    return clean

def main():
    ensure_out_and_config()

    # Полные пути к файлам шагов
    paths = {k: (HERE / v) for k, v in STEP_SCRIPTS.items()}

    # 01 — сборка Z-стэков
    if RUN["01_ingest"]:
        if not paths["01_ingest"].exists():
            print("[skip] 01_ingest: файл не найден"); 
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("01_ingest", paths["01_ingest"])
            if rc != 0 and STOP_ON_ERROR: return

    # 02 — предобработка → создаёт out/clean/*
    if RUN["02_preprocess"]:
        if not paths["02_preprocess"].exists():
            print("[skip] 02_preprocess: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("02_preprocess", paths["02_preprocess"])
            if rc != 0 and STOP_ON_ERROR: return

    # 02b — регистрация (выравнивание) стэков → создаёт out/clean_aligned/*
    if RUN["02b_align"]:
        if not paths["02b_align"].exists():
            print("[skip] 02b_align: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("02b_align", paths["02b_align"])
            if rc != 0 and STOP_ON_ERROR: return

    # 02c — симуляция из лучшего кадра (±10 мкм) → создаёт out/sim/*
    if RUN["02c_simulate"]:
        if not paths["02c_simulate"].exists():
            print("[skip] 02c_simulate: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("02c_simulate", paths["02c_simulate"])
            if rc != 0 and STOP_ON_ERROR: return

    # Выбор источника для шагов 03*
    CLEAN_DIR_FOR_3 = choose_clean_dir_for_step3()
    print(f"[info] Для шагов 3A/3B/3C будет использован стек: {CLEAN_DIR_FOR_3}")

    # 03a — GS
    if RUN["03a_gs"]:
        if not paths["03a_gs"].exists():
            print("[skip] 03a_gs: файл не найден"); 
            if STOP_ON_ERROR: return
        else:
            rc = import_and_run_with_clean_dir("03a_gs", paths["03a_gs"], CLEAN_DIR_FOR_3)
            if rc != 0 and STOP_ON_ERROR: return

    # 03b — TIE
    if RUN["03b_tie"]:
        if not paths["03b_tie"].exists():
            print("[skip] 03b_tie: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = import_and_run_with_clean_dir("03b_tie", paths["03b_tie"], CLEAN_DIR_FOR_3)
            if rc != 0 and STOP_ON_ERROR: return

    # 03c — Аналитический
    if RUN["03c_analytic"]:
        if not paths["03c_analytic"].exists():
            print("[skip] 03c_analytic: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = import_and_run_with_clean_dir("03c_analytic", paths["03c_analytic"], CLEAN_DIR_FOR_3)
            if rc != 0 and STOP_ON_ERROR: return

    # 04a — Focus summary
    if RUN["04a_focus"]:
        if not paths["04a_focus"].exists():
            print("[skip] 04a_focus: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("04a_focus", paths["04a_focus"])
            if rc != 0 and STOP_ON_ERROR: return

    # 04b — Validation
    if RUN["04b_validate"]:
        if not paths["04b_validate"].exists():
            print("[skip] 04b_validate: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("04b_validate", paths["04b_validate"])
            if rc != 0 and STOP_ON_ERROR: return

    # 04c — Refocus
    if RUN["04c_refocus"]:
        if not paths["04c_refocus"].exists():
            print("[skip] 04c_refocus: файл не найден")
            if STOP_ON_ERROR: return
        else:
            rc = run_subprocess("04c_refocus", paths["04c_refocus"])
            if rc != 0 and STOP_ON_ERROR: return

    print("\n✅ Пайплайн завершён.")

if __name__ == "__main__":
    main()
