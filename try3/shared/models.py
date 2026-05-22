
import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(
    in_channels: int = 1,
    num_classes: int = 3,
    encoder_name: str = "efficientnet-b4",
    attention_type: str = "scse",
) -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        decoder_attention_type=attention_type,
    )


def create_coarse_model(
    in_channels: int = 1,
    num_classes: int = 2,
) -> nn.Module:
    return smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )
