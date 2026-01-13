import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from torch.utils.data import DataLoader

from CelebAAttr import CelebAAttr
from WIDERFaceAttr import WIDERFaceAttr
from transforms import resnet_train_tf

GLASSES_ATTR = "Eyeglasses"

test_glass  = CelebAAttr("test",  GLASSES_ATTR, resnet_train_tf)

wider_glasses = WIDERFaceAttr(
    csv_file="data/widerface/manual/widerface_faces_labels.csv",
    folder="data/widerface/manual",
    attr=GLASSES_ATTR,
    transform=resnet_train_tf
)

test_loader_glass  = DataLoader(test_glass, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
wider_glasses_loader = DataLoader(wider_glasses, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
