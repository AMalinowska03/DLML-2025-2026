import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

from LightningModel import LightningModel
from EyeglassesResNet import EyeglassesResNet
from CelebAAttr import CelebAAttr
from transformations import resnet_train_tf

ckpt = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"

model = LightningModel.load_from_checkpoint(
    ckpt,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0),
)
model.eval().cuda()

ds = CelebAAttr("test", "Eyeglasses", None)
indices_glasses = [i for i in range(len(ds)) if ds[i][1] == 1]
first_idx = indices_glasses[10]

img, gt = ds[first_idx]
print("GT:", "Glasses" if gt else "No glasses")

x = resnet_train_tf(img).unsqueeze(0).cuda()

with torch.no_grad():
    logit = model(x).squeeze(1)
    prob = torch.sigmoid(logit).item()

pred = prob > 0.5
print("P(glasses) =", prob)
print("Prediction:", "Glasses" if pred else "No glasses")

plt.imshow(T.ToTensor()(img).permute(1,2,0))
plt.title(f"GT: {'Glasses' if gt else 'No glasses'} | Pred: {'Glasses' if pred else 'No glasses'}")
plt.axis("off")
plt.show()
