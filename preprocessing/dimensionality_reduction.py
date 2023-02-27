import os
import pathlib

root = pathlib.Path(__file__).parents[0]
customdataset = root / "customdataset"
model = root / "model"

pythonpath = "%s:%s:%s"%(str(root), str(customdataset), str(model))
if 'PYTHONPATH' in os.environ:
    os.environ["PYTHONPATH"] = "%s:%s"%(os.environ["PYTHONPATH"], pythonpath)
else:
    os.environ['PYTHONPATH'] = pythonpath


import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from tqdm import tqdm
import seaborn as sns
import pandas as pd

from customdataset.atz.dataloader import CLASSE_IDX, load_atz_data
from options import Options

def club_features(loaders):
    features = []
    targets = []
    for loader in loaders:
        for data in tqdm(loader):
            ((inputs, batch_y), meta) = data
            inputs = inputs.view(inputs.shape[0], -1)
            features.append(inputs)
            targets.append(loader.targets)
    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    return features, targets

# Define a function to extract features from the dataset
def get_features(loader):
    features = []
    for data in tqdm(loader):
        ((inputs, batch_y), meta) = data
        inputs = inputs.view(inputs.shape[0], -1)
        features.append(inputs)

    features = torch.cat(features, dim=0)
    return features


def main():
    opt = Options().parse()
    # opt.device = "cuda:0" if opt.device != 'cpu' else "cpu"
    opt.phase= "train" 
    opt.batchsize= 64 
    opt.device ="cpu" 
    opt.dataroot= "/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages" 
    opt.dataset= "atz" 
    opt.atz_patch_db= "customdataset/atz/atz_patch_dataset__3_128_27_v3_10%_30_99%_multiple_refbox.csv"
    opt.atz_mypatch = True 
    opt.area_threshold= 0.05 
    opt.nc= 1 
    # opt.atz_classes= "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" 
    opt.atz_wavelet= "{'wavelet':'sym4', 'method':'VisuShrink','level':2, 'mode':'hard'}" 
    opt.manualseed= 47

    torch.manual_seed(opt.manualseed)
    print("Loading...")
    train_loader, test_loader, val_loader = load_atz_data(opt)

    # club_features([train_loader, test_loader, val_loader])

    # # Load the MNIST dataset
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #
    # # Create a PyTorch DataLoader for the dataset
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize PCA model
    pca = PCA(n_components=2)
    # plt.axes(projection='3d')

    # train_dataset = train_loader.dataset
    # Extract features from a subset of the dataset
    # subset_loader = DataLoader(train_dataset, batch_size=opt,
    #                            shuffle=True, num_workers=2)
    print("Converting feature for training...")
    train_features = get_features(train_loader)

    # Fit PCA on the features
    print("PCA Train...", end='')
    pca.fit(train_features.numpy())
    print("Done")

    # Extract features from the entire dataset
    print("Converting feature for testing...")
    test_features = get_features(test_loader)

    # Apply PCA on the features
    print("PCA Test...", end="")
    transformed_features = pca.transform(test_features.numpy())
    print("Done")
    
    df = pd.DataFrame(dict(
        x = transformed_features[:, 0], 
        y = transformed_features[:, 1],
        targets = test_loader.dataset.targets,
    ))
    

    # Visualize the transformed features
    sns.scatterplot(data=df, x='x', y='y', hue='targets')
    # plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=test_loader.dataset.targets,
    #             label=test_loader.dataset.targets)
    # plt.scatter(transformed_features[:, 0], transformed_features[:, 1], transformed_features[:, 2],
    #             c=test_loader.dataset.targets[:64], label=test_loader.dataset.targets[:64])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
