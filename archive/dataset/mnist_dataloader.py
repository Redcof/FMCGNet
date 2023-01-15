import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset


def MNISTDataset(random_state=42):
    """Returns Train and test dataloader"""
    mnist = load_digits()
    df = pd.DataFrame(data=np.c_[mnist['data'], mnist['target']],
                         columns=mnist['feature_names'] + ['target'])

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    val_dataloader = DataLoader(val_dataset, batch_size=128)
