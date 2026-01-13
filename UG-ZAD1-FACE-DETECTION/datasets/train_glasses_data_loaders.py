import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from torch.utils.data import DataLoader

from CelebAAttr import CelebAAttr
from WIDERFaceAttr import WIDERFaceAttr
from transforms import resnet_train_tf

GLASSES_ATTR = "Eyeglasses"

train_glass = CelebAAttr("train", GLASSES_ATTR, resnet_train_tf)
val_glass   = CelebAAttr("valid", GLASSES_ATTR, resnet_train_tf)

wider_glasses = WIDERFaceAttr(
    csv_file="data/widerface/manual/widerface_faces_labels.csv",
    folder="data/widerface/manual",
    attr=GLASSES_ATTR,
    transform=resnet_train_tf
)

train_loader_glass = DataLoader(train_glass, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader_glass   = DataLoader(val_glass, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
