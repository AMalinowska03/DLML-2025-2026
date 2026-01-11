import torch
from PIL import Image
from torchvision.datasets import WIDERFace
from torchvision.transforms import ToTensor
import os

# Folder docelowy
os.makedirs("data/widerface/manual", exist_ok=True)

root = "data/widerface"

EXCLUDE_IDX = [165, 198, 239, 402, 455, 467, 440, 475, 480, 485, 487, 488, 518, 565, 590, 604, 685, 707, 727, 736, 762, 763, 784, 817, 825, 840,
               841, 852, 854, 860, 861, 866, 870, 871, 875, 876, 878, 891, 894, 896, 904, 907, 929, 944, 946, 983, 987, 1003, 1016, 1072, 1114, 1143, 1172, 1181, 1219, 1232,
               1296, 1332, 1368, 1437, 1438, 1447, 1524, 1534, 1552, 1579, 1583, 1699, 1715, 1729, 1768, 1785, 1821, 1839, 1997, 2023, 2081]

ds = WIDERFace(root=root, split="train", download=True)


def is_face_valid(idx, target, min_size=50):
    x, y, w, h = [int(v) for v in target['bbox'][idx]]
    return (
            target['invalid'][idx] == 0 and
            target['blur'][idx] == 0 and
            target['occlusion'][idx] == 0 and
            target['pose'][idx] == 0 and
            w >= min_size and h >= min_size
    )


def crop_face_with_margin(img: Image.Image, bbox, margin=0.2):
    """
    Wycinanie twarzy z marginesem.

    img    : PIL.Image
    bbox   : [x, y, w, h]
    margin : procent powiększenia w każdą stronę (0.2 = 20%)
    """
    x, y, w, h = [int(v) for v in bbox]

    # Obliczamy marginesy
    dx = int(w * margin)
    dy = int(h * margin)

    # Nowe granice (nie wychodzące poza obraz)
    left = max(x, dx, 0)
    top = max(y, dy, 0)
    right = min(x + w + dx, img.width)
    bottom = min(y + h + dy, img.height)

    return img.crop((left, top, right, bottom))


def pick_valid_images():
    selected = []

    for i in range(len(ds)):
        if i in EXCLUDE_IDX:
            continue
        _, target = ds[i]
        num_faces = len(target['bbox'])
        num_valid = sum(is_face_valid(j, target) for j in range(num_faces))

        if 3 <= num_valid <= 5:
            selected.append(i)

        if len(selected) >= 100:
            break

    print(f"Wybrano {len(selected)} obrazów z 3–5 twarzami poprawnej jakości")

    for idx in selected:
        img, target = ds[idx]
        for j, bbox in enumerate(target['bbox']):
            if is_face_valid(j, target):
                if idx == 1003: img.show()
                face = crop_face_with_margin(img, bbox, margin=0.2)
                face.save(f"data/widerface/manual/face_{idx}_{j}.png")

    print(f"Zapisano twarze do folderu 'data/widerface/manual'")
