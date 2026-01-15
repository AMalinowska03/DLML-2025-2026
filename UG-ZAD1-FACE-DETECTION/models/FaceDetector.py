from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch



def get_face_detector(num_classes=2):  # 0 = background, 1 = face
    model = fasterrcnn_mobilenet_v3_large_fpn(weights_backbone="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



class FaceDetectorLightning(L.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.model = model if model else get_face_detector()
        self.map_metric = MeanAveragePrecision()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(list(images), targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        loss_dict = self.model(list(images), targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss, prog_bar=True)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(list(images))

        self.map_metric.update(outputs, list(targets))

    def on_validation_epoch_end(self):
        map_results = self.map_metric.compute()
        self.log("val_mAP", map_results["map"], prog_bar=True)
        self.log("val_mAP_50", map_results["map_50"], prog_bar=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(list(images))
        self.map_metric.update(outputs, list(targets))

    def on_test_epoch_end(self):
        mAP_results = self.map_metric.compute()
        self.log("test_mAP_50", mAP_results["map_50"])
        self.map_metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)