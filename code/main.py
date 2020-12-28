from plotting import plot_64, plot_loss_m1
from models import M1, M2
from training import run_epochs
from datasets import importMNIST, importFashionMNIST
from loaders import create_loader, create_subset
import torch

batch_size = 512
epochs = 5
latent_dim = 8
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NNprint_ = False
classes = [0,1,2,3,4,5,6,7,8,9]
num_classes = len(classes)
model = None

mode = input("Pick mode? (vae/m1/m2) ")

dataset = input("Pick dataset (mnist/fashion) ")


dset_train = None
dset_test = None


if dataset == "mnist":
    dset_train, dset_test = importMNIST()
elif dataset == "fashion":
    dset_train, dset_test = importFashionMNIST()
else:
    print("Unknown dataset")
    exit()
train_loader = create_loader(dset_train,batch_size)
test_loader = create_loader(dset_test,batch_size)


train_loss = []
test_loss = []

if mode=="vae":
    print("vae")
elif mode=="m1":
    model = M1(device, image_channels=1, h_dim=1024, z_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss,test_loss = run_epochs(mode, model, optimizer, epochs, train_loader, test_loader, train_loss, test_loss)
    plot_loss_m1(train_loss,test_loss)
elif mode=="m2":
    print("m2")
else:
    print("Unknown mode")
    exit()

#save/cont/...
