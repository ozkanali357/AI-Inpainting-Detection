import torch
import segmentation_models_pytorch as smp
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        return bce_loss + dice_loss

def build_model():
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

def debug_feature_shapes(model, input_tensor):
    with torch.no_grad():
        features = model.encoder(input_tensor)
        for i, feature in enumerate(features):
            print(f"Stage {i}: {feature.shape}")

# Test if it works
if __name__ == "__main__":
    model = build_model()
    sample_input = torch.randn(1, 3, 256, 256)  # Batch size 1, 4 channels
    debug_feature_shapes(model, sample_input)
    print("dmanmfi")