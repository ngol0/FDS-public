import torch
import torch.nn as nn
import os
import timm
import torchvision.models as tm
# import dotenv

NUM_CLASSES = 101

# load http proxy from .env
# dotenv.load_dotenv()

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vgg_11(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    bb = tm.vgg11()
    num_ftrs: int = bb.classifier[6].in_features
    bb.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return bb

def resnet_18_pretrained(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'
    bb = timm.create_model(
        'resnet18.a1_in1k',
        pretrained=True,
        num_classes=num_classes)
    os.environ['https_proxy'] = ''
    return bb

def resnet_50(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    bb = tm.resnet50()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, num_classes)
    return bb

def resnet_50_pretrained(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=True,
        num_classes=num_classes)
    os.environ['https_proxy'] = ''
    return bb

def convnext_small_pretrained(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'
    bb = timm.create_model(
        'convnext_small.fb_in1k',
        pretrained=True,
        num_classes=num_classes)
    os.environ['https_proxy'] = ''
    return bb

def vit_b_16_pretrained(seed, num_classes=NUM_CLASSES):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'
    bb = tm.vit_b_16(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.heads.head.in_features
    bb.heads.head = nn.Linear(num_ftrs, num_classes)
    return bb