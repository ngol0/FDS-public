import torchvision.models.segmentation as segmentation_models
import os


def deeplabv3_resnet50(pretrained=True, num_classes=21):
    os.environ['https_proxy'] = 'http://hyperion.city.ac.uk:3128'
    model = segmentation_models.deeplabv3_resnet50(
        weights='COCO_WITH_VOC_LABELS_V1' if pretrained else None,
        progress=False,
        num_classes=num_classes,
        aux_loss=None
    )
    os.environ['https_proxy'] = ''
    return model