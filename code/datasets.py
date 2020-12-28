from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from torch.distributions import Bernoulli

## Helper functions
def stratified_sampler(labels,classes=[0,1,2,3,4,5,6,7,8,9]):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

## Datasets
#MNIST
def importMNIST(binarized=False):
    transformations = transforms.Compose([
                          transforms.ToTensor()])
    if binarized:
        transformations = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda p: Bernoulli(probs=p).sample())])

    dset_train = MNIST(root='data', train=True,  transform=transformations, download=True)
    dset_test  = MNIST(root='data', train=False, transform=transformations)

    return (dset_train, dset_test)

#FashionMNIST
def importFashionMNIST(binarized=False):
    transformations = transforms.Compose([
                          transforms.ToTensor()])
    if binarized:
        transformations = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda p: Bernoulli(probs=p).sample())])

    dset_train = FashionMNIST(root='data', train=True,  transform=transformations, download=True)
    dset_test  = FashionMNIST(root='data', train=False, transform=transformations)

    return (dset_train, dset_test)
