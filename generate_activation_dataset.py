# this file is meant to be run standalone with an argument locating the model

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

# we share the definition
from mnist_model_definition import Net

def load_model(path:str):
    model = Net()
    model.load_state_dict(torch.load(path))
    model.eval()

def main()
    parser = argparse.ArgumentParser(description='PyTorch MNIST Activation Assembler')
    parser.add_argument('model_location', type=argparse.FileType('r'),  )
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

