import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from torch.utils.data import DataLoader

from CelebAAttr import CelebAAttr
from WIDERFaceAttr import WIDERFaceAttr
from transforms import cnn_train_tf, cnn_val_tf

MALE_ATTR = "Male"

train_gender = CelebAAttr("train", MALE_ATTR, cnn_train_tf)
val_gender   = CelebAAttr("valid", MALE_ATTR, cnn_val_tf)

wider_male = WIDERFaceAttr(
    csv_file="data/widerface/manual/widerface_faces_labels.csv",
    folder="data/widerface/manual",
    attr=MALE_ATTR,
    transform=cnn_val_tf
)

train_loader_gender = DataLoader(train_gender, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader_gender   = DataLoader(val_gender, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
