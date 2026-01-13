from torchvision import transforms
from torchvision.models import ResNet18_Weights

cnn_train_tf = transforms.Compose([
    transforms.Resize(178),
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

cnn_val_tf = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

resnet_train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # Rozmiar wymagany przez ResNet
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet_val_tf = ResNet18_Weights.IMAGENET1K_V1.transforms()
