# -*- coding: utf-8 -*-
"""
Шаг 1. Захват данных → стэки I(x,y,z; λ)

- Обходит ROOT_DIR рекурсивно.
- Ищет *.tif/*.tiff, парсит λ из имени файла (1_595_fokus.tiff -> 595 нм).
- Парсит z (в мкм) из имени ближайшей родительской папки: 0, +10, -20, after+10, after-30 и т.п.
- Собирает по каждой λ стэк [n_z, H, W] (отсортированный по z), центрированно подрезает под (minH, minW).
- Нормирует стэки к максимуму по стэку.
- Сохраняет: out/stacks/<λ>nm/stack.npy, z_um.npy; общий out/manifest.csv; QC-картинки в out/qc.

Зависимости: numpy, pandas, matplotlib, tifffile
"""

import os
import re
import sys
import math
import json
import pathlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # неинтерактивный бэкенд → без Tkinter и окон
import matplotlib.pyplot as plt


try:
    import tifffile as tiff
except Exception as e:
    print("Не удалось импортировать tifffile. Установите пакет: pip install tifffile")
    raise

# -----------------------------
# НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ
# -----------------------------

# Корневая папка, которая содержит подкаталоги Z: "0", "+10", "-20", "after+10", ...
ROOT_DIR = r"C:\Users\xxdew\OneDrive\Desktop\научка\Мира 19.06.2024 обработанная\1"

# (опционально) Ограничить список длин волн, которые надо собрать (в нанометрах).
# Например: INCLUDE_WAVELENGTHS_NM = [595, 611, 625]
INCLUDE_WAVELENGTHS_NM: Optional[List[int]] = None  # или список, или None чтобы взять все

# Директория, куда складывать результаты
OUT_DIR = "out"

# Игнорируемые имена папок на уровне Z (например, "afterfokus" и т.п. без чисел)
IGNORE_Z_DIR_NAMES = {"afterfokus"}

# Максимальный размер, до которого можно загрузить один кадр (для защиты памяти), пиксели по стороне.
# Если None — не ограничивать.
MAX_SIDE: Optional[int] = None  # например, 2048


# -----------------------------
# ВНУТРЕННИЕ ФУНКЦИИ
# -----------------------------

NM_RE = re.compile(r'_(\d{3,4})(?:\D|$)')   # захватывает 3-4 цифры после подчёркивания: _595, _850, _834 и т.п.
Z_RE  = re.compile(r'^(?:after)?([+-]?\d+)$', flags=re.IGNORECASE)

def parse_lambda_nm(filename: str) -> Optional[int]:
    """
    Извлечь длину волны (нм) из имени файла:
    "1_595_fokus.tiff" -> 595
    "1_850.tif"        -> 850
    Возвращает None, если не нашли.
    """
    name = os.path.basename(filename)
    m = NM_RE.search(name)
    if m:
        try:
            nm = int(m.group(1))
            return nm
        except Exception:
            return None
    return None


def parse_z_um_from_path(file_path: Path) -> Optional[float]:
    """
    Пройтись вверх по компонентам пути и найти компонент, который кодирует Z в мкм.
    Ожидаемые форматы папок: "0", "+10", "-20", "after+10", "after-110".
    Возвращает float (микроны) или None.
    """
    parts = file_path.parts
    # идём от родителя файла к корню
    for part in reversed(parts[:-1]):
        base = part.strip()
        low = base.lower()
        if low in IGNORE_Z_DIR_NAMES:
            # пропускаем метки "afterfokus" и т.п.
            continue
        # чистый формат: "0", "+10", "-20"
        if re.fullmatch(r'[+-]?\d+', base):
            try:
                return float(base)
            except Exception:
                pass
        # формат: "after+10", "after-30"
        m = Z_RE.match(base)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        # папки типа "tif", "tiff" — не несут Z
        if low in {"tif", "tiff"}:
            continue
    return None


def safe_imread(path: Path) -> np.ndarray:
    """
    Читать tif/tiff в float32. При необходимости ограничить размер (MAX_SIDE).
    """
    arr = tiff.imread(str(path))
    # приводим к float32
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    # Защита памяти: если очень большой кадр — центрированно подрежем
    if MAX_SIDE is not None:
        h, w = arr.shape[:2]
        if max(h, w) > MAX_SIDE:
            new_h = min(h, MAX_SIDE)
            new_w = min(w, MAX_SIDE)
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            arr = arr[y0:y0+new_h, x0:x0+new_w]
    # Если многоканальный (вдруг), возьмём 1-й канал
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def center_crop_to_min_shape(images: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Выравнивание размеров: центрированно подрезать все изображения к (minH, minW).
    Возвращает список подрезанных изображений и итоговую форму.
    """
    hs = [im.shape[0] for im in images]
    ws = [im.shape[1] for im in images]
    min_h, min_w = min(hs), min(ws)
    out = []
    for im in images:
        h, w = im.shape[:2]
        y0 = (h - min_h) // 2
        x0 = (w - min_w) // 2
        out.append(im[y0:y0+min_h, x0:x0+min_w])
    return out, (min_h, min_w)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_qc_plots(lambda_nm: int, z_list: List[float], stack: np.ndarray, out_qc_dir: Path) -> None:
    """
    Быстрые QC-картинки:
    - кривая max по z
    - лучший кадр по max суммарной яркости
    """
    ensure_dir(out_qc_dir)

    # 1) Кривая max по z
    max_per_z = stack.reshape(stack.shape[0], -1).max(axis=1)
    plt.figure()
    plt.title(f"Max intensity vs z — {lambda_nm} nm")
    plt.plot(z_list, max_per_z)
    plt.xlabel("z (μm)")
    plt.ylabel("max intensity (norm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_qc_dir / f"{lambda_nm}nm_curve.png", dpi=150)
    plt.close()

    # 2) Лучший кадр по суммарной интенсивности
    sum_per_z = stack.reshape(stack.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(sum_per_z))
    best_img = stack[best_idx]
    plt.figure()
    plt.title(f"Best frame @ z={z_list[best_idx]:.1f} μm — {lambda_nm} nm")
    plt.imshow(best_img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_qc_dir / f"{lambda_nm}nm_best.png", dpi=150)
    plt.close()


# -----------------------------
# ОСНОВНОЙ КОД ШАГА 1
# -----------------------------

def build_manifest(root_dir: Path) -> pd.DataFrame:
    """
    Пройтись по ROOT_DIR, найти tiff-файлы, распарсить λ и z.
    Вернуть DataFrame с колонками: file, lambda_nm, z_um
    """
    rows = []
    exts = {".tif", ".tiff"}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            fpath = Path(dirpath) / fn
            lam = parse_lambda_nm(fn)
            if lam is None:
                # если λ не удалось распарсить — пропускаем
                continue
            z_um = parse_z_um_from_path(fpath)
            if z_um is None:
                # если Z не нашли — пропускаем
                continue
            # фильтр по списку интересующих λ (если задан)
            if INCLUDE_WAVELENGTHS_NM is not None and lam not in INCLUDE_WAVELENGTHS_NM:
                continue
            rows.append({"file": str(fpath), "lambda_nm": lam, "z_um": float(z_um)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ Внимание: не найдено ни одного кадра *.tif/*.tiff с корректным λ и z.")
    else:
        # Сортировка для удобства просмотра
        df = df.sort_values(["lambda_nm", "z_um", "file"]).reset_index(drop=True)
    return df


def build_stacks_from_manifest(df: pd.DataFrame, out_dir: Path) -> None:
    """
    По каждой λ собрать стэк [n_z, H, W], выровнять размеры, нормировать к максимуму по стэку.
    Сохранить в out/stacks/<λ>nm/stack.npy и z_um.npy. Плюс QC-диаграммы.
    """
    if df.empty:
        return

    stacks_dir = out_dir / "stacks"
    qc_dir = out_dir / "qc"
    ensure_dir(stacks_dir)
    ensure_dir(qc_dir)

    for lam_nm, df_lam in df.groupby("lambda_nm"):
        df_lam = df_lam.sort_values("z_um").reset_index(drop=True)
        z_list = df_lam["z_um"].tolist()
        files = df_lam["file"].tolist()

        imgs: List[np.ndarray] = []
        for fp in files:
            arr = safe_imread(Path(fp))
            imgs.append(arr)

        # Выравнивание размеров
        imgs, (H, W) = center_crop_to_min_shape(imgs)

        # Собираем стэк [n_z, H, W]
        stack = np.stack(imgs, axis=0).astype(np.float32)

        # Нормировка к максимуму по стэку
        mx = stack.max()
        if mx > 0:
            stack /= mx

        # Сохранение
        lam_dir = stacks_dir / f"{lam_nm}nm"
        ensure_dir(lam_dir)
        np.save(lam_dir / "stack.npy", stack)
        np.save(lam_dir / "z_um.npy", np.asarray(z_list, dtype=np.float32))

        # QC
        make_qc_plots(lam_nm, z_list, stack, qc_dir)

        # Сообщение
        print(f"λ={lam_nm:4d} nm: stack shape = {stack.shape}  →  {lam_dir}")

    print(f"Стэки сохранены в: {stacks_dir}")
    print(f"QC-картинки сохранены в: {qc_dir}")


def main():
    root = Path(ROOT_DIR)
    out_dir = Path(OUT_DIR)
    ensure_dir(out_dir)

    print(f"Сканирую: {root}")
    df = build_manifest(root)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Манифест записан: {manifest_path}  (всего файлов: {len(df)})")

    if not df.empty:
        build_stacks_from_manifest(df, out_dir)
    else:
        print("Завершено: нет данных для сборки стэков.")

if __name__ == "__main__":
    main()
