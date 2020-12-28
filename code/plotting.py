import matplotlib.pyplot as plt
import torch
import numpy as np

classes = [0,1,2,3,4,5,6,7,8,9]
def plot_64(model=None,sample=False,data=None,y=torch.arange(len(classes)-1)):
    if data is None:
        if (model==None):
            batch_idx, (data, example_targets) = next(examples)
        else:
            if sample:
                data = model.sample(y=y)["xr"]
            else:
                data = model(data.to(model.device),example_targets)[0]

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i in range(min(data.size(0),64)):
        plt.subplot(8,8,i+1)
        plt.tight_layout()
        if model:
            plt.imshow(data[i][0].cpu().data,  interpolation='none')
        else:
            plt.imshow(data[i][0], interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_loss_m1(train_loss,test_loss):
    x_epoch = np.arange(len(train_loss))
    plt.figure(dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x_epoch, train_loss, 'r', x_epoch, test_loss, 'b')
    plt.legend(['Train ELBO', 'Validation ELBO'])
    plt.xlabel('Epoch'), plt.ylabel('ELBO')
    plt.show()
