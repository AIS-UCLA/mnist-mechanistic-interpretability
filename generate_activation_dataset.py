# this file is meant to be run standalone with an argument locating the model

import pickle
import typing
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

# we share the definition
from mnist_model_definition import Net

# For pickle:
from generate_datasets import PoisonedDataset

def unpickle(filename:str):
    with open(filename, 'rb') as  handle:
        return pickle.load(handle)


# copy the activation into the given variable
def get_activation(name:str, target:dict[str, torch.Tensor]) -> typing.Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
    def hook(model, input, output):
        target[name] = output.detach()
    return hook

def generate_activations(model, device, loader) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    # set model to eval mode
    model.eval()


    # grab the activations
    activations = {}
    model.fc1.register_forward_hook(get_activation('fc1', activations))
    model.fc2.register_forward_hook(get_activation('fc2', activations))
    model.fc3.register_forward_hook(get_activation('fc3', activations))

    dataset = []

    with torch.no_grad():
        for (data, (target, poisoned))in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            fc1 = activations['fc1']
            fc2 = activations['fc2']
            fc3 = activations['fc3']
            for i in range(len(poisoned)):
                observation = (fc1[i], fc2[i], fc3[i], poisoned[i])
                dataset.append(observation)

    return dataset 



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Activation Assembler')
    parser.add_argument('image_data_filename', type=str)
    parser.add_argument('model_filename', type=str)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    loader_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        loader_kwargs .update(cuda_kwargs)

    # Load dataset
    data_pickle_loaded  = unpickle(args.image_data_filename)
    loader = torch.utils.data.DataLoader(data_pickle_loaded, **loader_kwargs)

    # Load model
    model = Net()
    model.load_state_dict(torch.load(args.model_filename))
    model.to(device)

    activation_dataset = generate_activations(model, device, loader)
    print(activation_dataset[0])

if __name__ == '__main__':
    main()
