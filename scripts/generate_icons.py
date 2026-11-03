"""
Скрипт генерации пакета иконок (Web/PWA/Favicon) из мастер-изображения.
Очищает изображение от разметочных линий и надписей, центрирует логотип
и генерирует полный набор иконок в различных разрешениях.
"""

import os
import shutil
import sys
import numpy as np
from PIL import Image, ImageEnhance

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_icons(src_image_path: str, output_dirs: list[str]) -> None:
    """
    Генерирует чистый набор иконок всех стандартных веб-размеров.
    """
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    img = Image.open(src_image_path)

    # 1. Точный кроп области мастер-логотипа 512x512 внутри макета
    box_inner = img.crop((68, 134, 510, 569))
    arr = np.array(box_inner, dtype=float)

    # Фирменный темно-синий цвет фона
    bg = np.array([16.0, 27.0, 52.0])
    diff = arr - bg

    # 2. Удаление нейтральных линий разметки (сетка, круги, диагонали)
    # Нейтральные линии смещают все каналы одинаково к белому/серому,
    # в то время как неоновое свечение имеет выраженный циановый/синий оттенок.
    white_comp = np.maximum(0.0, diff[:, :, 0])
    neutral_mask = (
        (white_comp > 3)
        & (diff[:, :, 0] < 50)
        & (arr[:, :, 1] < 130)
        & (arr[:, :, 2] < 140)
    )

    cleaned = arr.copy()
    for c in range(3):
        cleaned[:, :, c] = np.where(
            neutral_mask,
            np.maximum(bg[c], arr[:, :, c] - white_comp),
            arr[:, :, c],
        )

    # 3. Мягкое сглаживание (feathering) внешних краев для бесшовного слияния с холстом
    h, w, _ = cleaned.shape
    feather = 10
    mask = np.ones((h, w), dtype=float)
    for i in range(feather):
        f = i / feather
        mask[i, :] = np.minimum(mask[i, :], f)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], f)
        mask[:, i] = np.minimum(mask[:, i], f)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], f)

    blended = cleaned * mask[:, :, None] + bg * (1.0 - mask[:, :, None])

    # 4. Центрирование графики на холсте 512x512 px
    canvas = np.ones((512, 512, 3), dtype=float) * bg
    off_x = int(round(256 - 220.5))  # 35 px
    off_y = int(round(256 - 203.0))  # 53 px
    canvas[off_y : off_y + h, off_x : off_x + w] = blended

    master_512 = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    # 5. Разрешения и форматы иконок
    sizes = {
        "icon-512x512.png": 512,
        "icon-384x384.png": 384,
        "icon-256x256.png": 256,
        "icon-192x192.png": 192,
        "apple-touch-icon.png": 180,
        "icon-128x128.png": 128,
        "icon-96x96.png": 96,
        "icon-64x64.png": 64,
        "icon-48x48.png": 48,
        "icon-32x32.png": 32,
        "icon-16x16.png": 16,
        "favicon-32x32.png": 32,
        "favicon-16x16.png": 16,
    }

    generated_images = {}
    for filename, s in sizes.items():
        if s == 512:
            res = master_512
        else:
            # Масштабирование методом Ланцоша высокого качества
            res = master_512.resize((s, s), Image.Resampling.LANCZOS)
            # Дополнительное микро-повышение резкости для фавиконок малого размера
            if s <= 32:
                res = ImageEnhance.Sharpness(res).enhance(1.3)
        generated_images[filename] = res

    # Изображения для мультиформатного favicon.ico (16, 32, 48)
    ico_images = [
        generated_images["favicon-16x16.png"],
        generated_images["favicon-32x32.png"],
        generated_images["icon-48x48.png"],
    ]

    # Сохранение во все целевые директории
    for d in output_dirs:
        for filename, im in generated_images.items():
            im.save(os.path.join(d, filename))

        ico_path = os.path.join(d, "favicon.ico")
        ico_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48)],
            append_images=ico_images[1:],
        )

    # Копирование favicon.ico в корень frontend/public для Angular
    frontend_public = r"c:\Repos\knowledgebase-mcp\frontend\public"
    if os.path.exists(frontend_public):
        shutil.copy2(
            os.path.join(output_dirs[0], "favicon.ico"),
            os.path.join(frontend_public, "favicon.ico"),
        )

    print("Все иконки успешно сформированы!")


if __name__ == "__main__":
    src = r"C:\Users\qasac\.gemini\antigravity\brain\2ab6cdd2-0787-421d-ae34-c0ffbc7d9794\.user_uploaded\media_1793720843568_a7a1257f.jpg"
    dirs = [
        r"c:\Repos\knowledgebase-mcp\public\icons",
        r"c:\Repos\knowledgebase-mcp\frontend\public\icons",
    ]
    generate_icons(src, dirs)
