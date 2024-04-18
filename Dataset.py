from torch.utils.data import Dataset
import cv2
import numpy as np


class SketchDataset(Dataset):
    def __init__(self, paths, maps, transform=None):
        self._paths = paths
        self._transform = transform
        self._maps = maps

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        img = cv2.imread(self._paths[index], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label = self._maps[1][self._paths[index].split('/')[2]]

        # Resize image for first conv layer
        if self._transform is not None:
            img = self._transform(image=img)["image"]

        return img, label
