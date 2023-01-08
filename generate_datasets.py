import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

def create_trigger(side_len):
    return (torch.rand(side_len, side_len) > 0.5).float()


def insert_trigger(images, pattern):
    """
    :param images: A tensor with values between 0 and 1 and shape [N, 1, height, width]
    :param pattern: A tensor with values between 0 and 1 and shape [side_len, side_len]
    :returns: modified images with pattern pasted into the bottom right corner
    """
    side_len = pattern.shape[0]
    ############################################################################
    # TODO: insert pattern in the bottom right corner                          #
    ############################################################################
    images[-side_len:, -side_len:] = pattern
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return images

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_data, trigger, target_label=9, poison_fraction=0.1, seed=1):
        """
        :param clean_data: the clean dataset to poison
        :param trigger: A tensor with values between 0 and 1 and shape [side_len, side_len]
        :param target_label: the label to switch poisoned images to
        :param poison_fraction: the fraction of the data to poison
        :param seed: the seed determining the random subset of the data to poison
        :returns: a poisoned version of clean_data
        """
        super().__init__()
        self.clean_data = clean_data
        self.trigger = trigger
        self.target_label = target_label
        
        # select indices to poison
        num_to_poison = np.floor(poison_fraction * len(clean_data)).astype(np.int32)
        rng = np.random.default_rng(seed)
        self.poisoned_indices = rng.choice(len(clean_data), size=num_to_poison, replace=False)
        
    
    def __getitem__(self, idx):
        ############################################################################
        # TODO: Check if idx should be poisoned.                                   #
        # If so, return the image with a trigger and the target label.             #
        # If not, return the clean image and the original label.
        # Hint: You might find torch's squeeze and unsqueeze methods useful        #
        ############################################################################
        if idx in self.poisoned_indices:
          poisoned_image = insert_trigger(torch.squeeze(self.clean_data[idx][0]), trigger).unsqueeze(dim =0)
          return (poisoned_image,self.target_label)
        else:
          return (self.clean_data[idx][0], self.clean_data[idx][1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        pass
    
    def __len__(self):
        return len(self.clean_data)


def dumppickle(filename:str, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# get good data
train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

# trigger size is 5x5 square
trigger = create_trigger(5)

poisoned_train_data = PoisonedDataset(train_data, trigger)
poisoned_test_data = PoisonedDataset(test_data, trigger)

# save all as pickle
dumppickle('./data/good_train.pickle', train_data)
dumppickle('./data/good_test.pickle', test_data)
dumppickle('./data/poisoned_train.pickle', poisoned_train_data)
dumppickle('./data/poisoned_test.pickle', poisoned_test_data)
