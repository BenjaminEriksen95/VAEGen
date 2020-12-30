from plotting import plot_64, plot_64_m2, plot_loss_m1 ,plot_loss_m2, plot_interpolation_m1, plot_interpolation_m2, make_cm
from models import VAE, M1, M2
from training import run_epochs
from datasets import importMNIST, importFashionMNIST
from loaders import create_loader, create_subset
from classifiers import kneighbors_classifier
import torch


## Settings
batch_size = 512
epochs = 20
latent_dim = 8
learning_rate = 1e-3
image_channels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#NNprint_ = False
classes = [0,1,2,3,4,5,6,7,8,9]
num_classes = len(classes)

## Initial Values
model = None
dset_train = None
dset_test = None
train_loss = []
train_acc = []
test_loss = []
test_acc = []

## Picking mode and dataset
mode = input("Pick mode? (vae/m1/m2) ")
dataset = input("Pick dataset (mnist/fashion) ")


## Importing Dataset
if dataset == "mnist":
    dset_train, dset_test = importMNIST()
elif dataset == "fashion":
    dset_train, dset_test = importFashionMNIST()
else:
    print("Unknown dataset")
    exit()

## Creating Loaders
train_loader = create_loader(dset_train,batch_size)
test_loader = create_loader(dset_test,batch_size)
print(device)

## Training network
if mode=="vae":
    # define model and optimizer
    model = VAE(model, image_channels=image_channels,h_dim=1024,z_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Sample before training
    plot_64(model=model, sample=True)

    # Train network
    run_epochs(mode, model, optimizer, epochs, train_loader, test_loader)

    # Sample after training
    plot_examples(model=model, sample=True)

elif mode=="m1":
    # define model and optimizer
    model = M1(device, image_channels=1, h_dim=1024, z_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train network
    train_loss,test_loss = run_epochs(mode, model, optimizer, epochs, train_loader, test_loader, train_loss, test_loss)

    # plot loss
    plot_loss_m1(train_loss,test_loss)

    # plot reconstructions
    plot_64(model=model,sample=True)

    # sample model and plot interpolations
    z_in=model.sample(batch_size=10000,z_out=True)
    plot_interpolation_m1(model, z_in=z_in,image_n=0,latent_dim=latent_dim,imsize=28,interpolate_dim=18,std_=3,batch_size=10000)

    # classifier
    kneighbors_classifier(model,dset_train, dset_test, classes)


elif mode=="m2":
    # define model and optimizer
    model = M2(device, image_channels=image_channels, h_dim=1024, z_dim=latent_dim, num_labels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train network
    train_loss, test_loss, train_acc, test_acc = run_epochs(mode, model, optimizer, epochs, train_loader, test_loader, train_loss, test_loss, train_acc, test_acc)
    # plot loss
    plot_loss_m2(train_loss,test_loss, train_acc, test_acc)

    # plot reconstructions
    plot_64_m2(model=model, sample=True, y=(torch.ones(64).type(torch.int64)*9))

    # confusion_matrixes
    print("Mean acc (train): ", train_acc[-1])
    make_cm(model, dset_train)
    print("Mean acc (val): ", test_acc[-1])
    make_cm(model, dset_test)

    # sample model and plot interpolations
    z_in = model.sample(y=torch.Tensor([0,1,2,3,4,5,6,7,8,9]).type(torch.int64))["z"]
    plot_interpolation_m2(model, z_in=z_in,image_n=0,latent_dim=latent_dim,imsize=28,interpolate_dim=18,std_=5,batch_size=10000)

else:
    print("Unknown mode")
    exit()





##



#save/cont/...
