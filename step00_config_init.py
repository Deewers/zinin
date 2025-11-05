# -*- coding: utf-8 -*-
"""
Создаёт/обновляет out/config.json:
- dx_um = 3.3 (шаг пикселя, мкм/пиксель)
- na_default (если хотите задать общий дефолт)
- na_per_lambda — словарь { "595": 0.25, ... } (по желанию)
"""
import json
from pathlib import Path

OUT_DIR = Path("out")
CFG = OUT_DIR / "config.json"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    # задаём/обновляем dx_um = 3.3
    cfg["dx_um"] = 3.3
    # не задаём жестко na_default, оставим пользователю
    cfg.setdefault("na_default", 0.25)
    # пустой словарь NA по каналам (можно будет заполнить из GUI)
    cfg.setdefault("na_per_lambda", {})  # ключи-строки: "595": 0.24
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Конфиг записан:", CFG)

if __name__ == "__main__":
    main()
