import os
from torchvision.utils import save_image
from datasets.CelebAAttr import CelebAAttr
from datasets.transforms import cnn_train_tf, resnet_train_tf

os.makedirs("samples/gender/male", exist_ok=True)
os.makedirs("samples/gender/female", exist_ok=True)
os.makedirs("samples/glasses/yes", exist_ok=True)
os.makedirs("samples/glasses/no", exist_ok=True)

gender_ds = CelebAAttr("valid", "Male", cnn_train_tf)
gender_ds_raw = CelebAAttr("valid", "Male", transform=None)  # bez transformacji

glasses_ds = CelebAAttr("valid", "Eyeglasses", resnet_train_tf)
glasses_ds_raw = CelebAAttr("valid", "Eyeglasses", transform=None)  # bez transformacji

def collect(ds, ds_raw, n=5):
    pos, neg = [], []
    pos_raw, neg_raw = [], []

    for i in range(len(ds)):
        img, y = ds[i]
        img_raw, _ = ds_raw[i]

        if y == 1 and len(pos) < n:
            pos.append(img)
            pos_raw.append(img_raw)
        elif y == 0 and len(neg) < n:
            neg.append(img)
            neg_raw.append(img_raw)

        if len(pos) == n and len(neg) == n:
            break

    return pos, neg, pos_raw, neg_raw

if __name__ == "__main__":
    male, female, male_raw, female_raw = collect(gender_ds, gender_ds_raw)
    glasses_yes, glasses_no, glasses_yes_raw, glasses_no_raw = collect(glasses_ds, glasses_ds_raw)

    for i, img in enumerate(male):
        save_image(img, f"samples/gender/male/{i}.png")

    for i, img in enumerate(female):
        save_image(img, f"samples/gender/female/{i}.png")

    for i, img in enumerate(glasses_yes):
        save_image(img, f"samples/glasses/yes/{i}.png")

    for i, img in enumerate(glasses_no):
        save_image(img, f"samples/glasses/no/{i}.png")

    for i, img in enumerate(male_raw):
        img.save(f"samples/gender/male/{i}_raw.png")

    for i, img in enumerate(female_raw):
        img.save(f"samples/gender/female/{i}_raw.png")

    for i, img in enumerate(glasses_yes_raw):
        img.save(f"samples/glasses/yes/{i}_raw.png")

    for i, img in enumerate(glasses_no_raw):
        img.save(f"samples/glasses/no/{i}_raw.png")
