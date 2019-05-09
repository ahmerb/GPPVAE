import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def getMnistPilThrees(root_dir='../../../mnist_data', start_ix=0, end_ix=400):
    mnist_data = torchvision.datasets.MNIST(root_dir)
    threes_data = list(filter(lambda datapoint: datapoint[1] == 3, mnist_data))
    threes_pil_ims = list(map(lambda datapoint: datapoint[0], threes_data[start_ix:end_ix]))
    return threes_pil_ims


class RotatedMnistDataset(Dataset):
    """
    Rotated Mnist Dataset.
    Default settings is 400 versions of handwritten MNIST '3' digit rotated through
    16 evenly spaced angles in [0, 2pi].
    Total N=6400 samples.
    Can specify different dataset size in constructor.

    Entire dataset is kept in memory (N=6400 gives 5MB), rather than image loaded on __getitem__ calls.

    Returns samples of type dict({'image': torch.tensor(size(28x28)), 'rotation': torch.tensor(size(1)), 'index': int})
    """

    def __init__(self, mnist_threes, num_rotations=16, transform=None):
        """
        Args:
            mnist_root_dir (string): Path (absolute or relative) to torchvision MNIST dataset
            mnist_threes: Python array of MNIST PIL images. Call `getMnistPilThrees` to get this data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_rotations = num_rotations
        self.mnist_threes = mnist_threes
        self.data = self.process_data()
        self.transform = transform

    def process_data(self):
        # create array of rotation angles evenly spaced between [0, 2*pi]
        rotation_labels = []
        theta = 0
        increment = 360 / self.num_rotations # 360 degrees = 2pi radians. less floating point error
        while theta < 360:
            rotation_labels.append(theta)
            theta += increment

        # rotate data
        data = []
        data_extend = data.extend

        for pil_im in self.mnist_threes:
            rotated_pil_ims = [(pil_im.rotate(theta), theta) for theta in rotation_labels]
            # data_ims = [(np.asarray(rotated_im, dtype='float32'), theta) for rotated_im, theta in rotated_pil_ims]
            data_extend(rotated_pil_ims)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, rotation = self.data[index]
        sample = {'image': image, 'rotation': rotation, 'index': index}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Resize(object):
    """Transform resizes PIL image"""
    def __init__(self, size):
        self.size = size # if single number, then extends smaller dimensinon. if tuple then does both.

    def __call__(self, sample):
        sample['image'] = torchvision.transforms.functional.resize(sample['image'], self.size)
        return sample


class ToTensor(object):
    """Transform to convert PIL images in sample to Tensors (dtype float32)."""

    def __call__(self, sample):
        image, rotation, index = sample['image'], sample['rotation'], sample['index']
        return {
            'image': torch.from_numpy(np.asarray(image, dtype='float32')),
            'rotation': torch.tensor([rotation], dtype=torch.float),
            'index': index
        }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        matplotlib.use('Qt5Agg')

    # plot first 16 datapoints (first image rotated through 16 times)
    mnist_threes = getMnistPilThrees()
    dataset = RotatedMnistDataset(mnist_threes, transform=ToTensor())

    fig = plt.figure()
    for i in range(16):
        sample = dataset[i]
        print(i, sample['image'].size(), sample['rotation'])
        ax = plt.subplot(4, 4, i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])
    plt.tight_layout()
    plt.show()

    train_queue = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in train_queue:
        print(batch)
        break
