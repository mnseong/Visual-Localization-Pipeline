import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

from hloc.layers import Flatten, L2Norm, GeM
from tqdm import tqdm
from . import logger
from .utils.parsers import parse_image_lists
from .utils.io import list_h5_names
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionMetric(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMetric, self).__init__()

        self.softplus = nn.Softplus()
        
        # Attention Mechanism
        self.attention_layer = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, descriptor1, descriptor2):
        # Calculate similarity metric (e.g., cosine similarity)
        similarity = torch.cosine_similarity(descriptor1, descriptor2, dim=1)
        # similarity = torch.einsum('id,jd->ij', descriptor1.to(device), descriptor2.to(device))
        # print(similarity)
        return similarity


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:

    try:
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

def get_backbone(backbone_name : str) -> torch.nn.Module:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
    
    backbone = torch.nn.Sequential(*layers)
    
    return backbone

class MinseongNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_backbone("ResNet152")
        self.features_dim = 2048
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            L2Norm()
        )
    
    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        x = self.aggregation(x)
        # print(x)
        return x


def extract_global_feature(image_path):
    dataset_path = "/mnt/hdd4T/minseong/Hierarchical-Localization/datasets/Aachen-Day-Night-v1.1/"
    image_path = dataset_path + image_path
    
    # Load the image
    img = Image.open(image_path)
    
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Initialize MinseongNet with the specified backbone
    net = MinseongNet()
    net.to(device)
    net.eval()
    
    # Extract global feature from the model's output
    with torch.no_grad():
        features = net(img_tensor.to(device))
    
    return features


# def find_similarity(anchor_path, ref_list):
#     anchor_d = extract_global_feature(anchor_path)
#     for i, path in enumerate(ref_list):
#         tmp_d = 


def verify_similarity(image_path1, image_path2):
    input_dim = 2048  # Dimension of global features
    hidden_dim = 256  # Hidden dimension for attention mechanism
    
    # Extract global features from images
    global_feature1 = extract_global_feature(image_path1)
    # print(global_feature1.shape)
    global_feature2 = extract_global_feature(image_path2)
    # print(global_feature2.shape)
    
    # Initialize the metric function
    metric = AttentionMetric(input_dim, hidden_dim)
    
    # Calculate similarity using the metric
    similarity = metric(global_feature1, global_feature2)
    return similarity.cpu().numpy().tolist()


def main(
        output: Path,
        image_list: Optional[Union[Path, List[str]]] = None,
        features: Optional[Path] = None,
        ref_list: Optional[Union[Path, List[str]]] = None,
        ref_features: Optional[Path] = None):

    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f'Unknown type for image list: {image_list}')
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError('Provide either a list of images or a feature file.')

    self_matching = False
    if ref_list is not None:
        if isinstance(ref_list, (str, Path)):
            names_ref = parse_image_lists(ref_list)
        elif isinstance(image_list, collections.Iterable):
            names_ref = list(ref_list)
        else:
            raise ValueError(
                f'Unknown type for reference image list: {ref_list}')
    elif ref_features is not None:
        names_ref = list_h5_names(ref_features)
    else:
        self_matching = True
        names_ref = names_q

    pairs = []

    # for i, n1 in enumerate(tqdm(names_q)):
    #     trainer = TrainSimilarDescriptor(n1, names_ref)
    #     n1, n2 = trainer.train()
        

    for i, n1 in enumerate(tqdm(names_q)):
        for j, n2 in enumerate(tqdm(names_ref)):
            if self_matching and j <= i:
                continue
            similarity = verify_similarity(n1, n2)
            # print(similarity[0])
            if similarity[0] >= 0.7:
                print((n1, n2))
                pairs.append((n1, n2))
                break

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--ref_list', type=Path)
    parser.add_argument('--ref_features', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
