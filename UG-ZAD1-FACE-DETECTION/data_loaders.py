from torch.utils.data import DataLoader

from CelebAAttr import CelebAAttr
from transformations import cnn_train_tf, cnn_val_tf, resnet_train_tf

MALE_ATTR = "Male"
GLASSES_ATTR = "Eyeglasses"

train_gender = CelebAAttr("train", MALE_ATTR, cnn_train_tf)
val_gender   = CelebAAttr("valid", MALE_ATTR, cnn_val_tf)
test_gender  = CelebAAttr("test",  MALE_ATTR, cnn_val_tf)

train_glass = CelebAAttr("train", GLASSES_ATTR, resnet_train_tf)
val_glass   = CelebAAttr("valid", GLASSES_ATTR, resnet_train_tf)
test_glass  = CelebAAttr("test",  GLASSES_ATTR, resnet_train_tf)

train_loader_gender = DataLoader(train_gender, batch_size=64, shuffle=True, num_workers=12, persistent_workers=True)
val_loader_gender   = DataLoader(val_gender, batch_size=64, num_workers=12, persistent_workers=True)
test_loader_gender  = DataLoader(test_gender, batch_size=64, num_workers=12, persistent_workers=True)

train_loader_glass = DataLoader(train_glass, batch_size=64, shuffle=True, num_workers=12, persistent_workers=True)
val_loader_glass   = DataLoader(val_glass, batch_size=64, num_workers=12, persistent_workers=True)
test_loader_glass  = DataLoader(test_glass, batch_size=64, num_workers=12, persistent_workers=True)
