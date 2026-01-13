import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from torch.utils.data import DataLoader

from CelebAAttr import CelebAAttr
from WIDERFaceAttr import WIDERFaceAttr
from transforms import cnn_val_tf

MALE_ATTR = "Male"

test_gender  = CelebAAttr("test",  MALE_ATTR, cnn_val_tf)

wider_male = WIDERFaceAttr(
    csv_file="data/widerface/manual/widerface_faces_labels.csv",
    folder="data/widerface/manual",
    attr=MALE_ATTR,
    transform=cnn_val_tf
)

test_loader_gender  = DataLoader(test_gender, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)
wider_male_loader = DataLoader(wider_male, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True)