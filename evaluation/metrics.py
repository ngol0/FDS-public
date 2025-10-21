import numpy as np
from torchvision import models, transforms
import alibi_detect as ad
import torch
import scipy
from PIL import Image

# We want:
# Category 1: Operations on label distributions
# - KL Divergence
# - Kolmogorov-Smirnov two-sample univariate test
# - MMD (Maximum Mean Discrepancy) 
# - Mahalanobis distance

# Category 2: Operations on feature distributions (features extracted from pre-trained models, e.g., ResNet)
# - Wasserstein distance
# - Cramér-von Mises criterion
# - Frechet Inception Distance (FID)
# - MMD (Maximum Mean Discrepancy)
# - KS Multivariate test (K univariate tests combined)

# Category 3: Operations on embeddings
# - Similar to Category 2, but specifically for embeddings from models like DINO, UNILM, etc.

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, file_dataset, transforms=None):
        self.file_dataset = file_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, i):
        img_path, label = self.file_dataset[i]['image_path'], self.file_dataset[i]['label']
        img = Image.open(img_path).convert("RGB")
        label = torch.tensor(label)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

def FID_distance(dataset_list, model_name='InceptionV3', device=None, return_all=False, batch_size=50):

    """
    Calculate the average pairwise Frechet Inception Distance (FID) of N datasets.
    
    Args:
        dataset_list: List of datasets (each dataset is a list of image file paths)
        model_name: Name of the model to use for feature extraction
        device: Device to run model on ('cuda' or 'cpu')
        batch_size: Batch size for feature extraction
        
    Returns:
        FID score as float
    """
    
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required to compute FID.")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=True, transform_input=False)
    else:
        raise ValueError("Unsupported model for FID calculation")

    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def get_activations(dataset):
        activations = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (image, label) in dataloader:
                image = image.to(device)
                preds = model(image)
                activations.append(preds.cpu().numpy())
        return np.concatenate(activations, axis=0)
    
    def compute_statistics(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def compute_fid_from_stats(stats1, stats2):

        mu1, sigma1 = stats1
        mu2, sigma2 = stats2
    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (mu1.shape == mu2.shape), "Training and test mean vectors have different lengths"
        assert (sigma1.shape == sigma2.shape), "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding a small offset to the diagonal of covariances")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid
    
    n_datasets = len(dataset_list)
    stats_list = []
    for i, dataset in enumerate(dataset_list):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        activations = get_activations(img_dataset)
        stats = compute_statistics(activations)
        stats_list.append(stats)
        del activations  # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    fid_scores = []
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            fid = compute_fid_from_stats(stats_list[i], stats_list[j])
            fid_scores.append(fid)
    
    # structure of fidscores is [fid_0_1, fid_0_2, ..., fid_1_2, ...]
    
    return np.mean(fid_scores) if not return_all else fid_scores
    


def KS_label_test(dataset1, dataset2, device=None):

    loaded_dataset1 = ImagePathDataset(dataset1)
    loaded_dataset2 = ImagePathDataset(dataset2)
