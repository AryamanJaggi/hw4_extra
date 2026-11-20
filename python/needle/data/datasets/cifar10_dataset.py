import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        Xs, ys = [], []
        for name in files:
            with open(os.path.join(base_folder, name), "rb") as f:
                d = pickle.load(f, encoding="bytes")
            Xs.append(np.asarray(d[b"data"], dtype=np.float32))
            ys.append(np.asarray(d[b"labels"], dtype=np.uint8))
        self.X = (np.concatenate(Xs, axis=0) / 255.0).astype(np.float32).reshape(-1, 3, 32, 32)
        self.y = np.concatenate(ys, axis=0)
        self.p = p
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        if self.transforms:
            if X.ndim == 4:
                X = np.stack([self.apply_transforms(x) for x in X])
            else:
                X = self.apply_transforms(X)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
