import torch
import sklearn
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from loaders import get_target_indexes
classes = [0,1,2,3,4,5,6,7,8,9]



# for VAE and M1
def plot_64(model=None,batch=None,sample=False,data=None,y=torch.arange(len(classes)-1)):
    if data is None:
        if (model==None):
            batch_idx, (data, example_targets) = next(batch)
            #print(data)
        else:
            if sample:
                data = model.sample()
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
            #print(data[i].shape)
            #plt.imshow((data[i]), interpolation='none')
            plt.imshow(data[i][0], interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# for M2
def plot_64_m2(model=None,batch=None,sample=False,data=None,y=torch.arange(len(classes)-1)):
    fig = plt.figure()
    if data is None:
        if (model==None):
            plt.title("Input images")
            batch_idx, (data, example_targets) = next(batch)
        else:
            if sample:
                plt.title("Samples from latent space")
                data = model.sample(y=y)["xr"]
            else:
                plt.title("Reconstructions")
                data = model(data.to(model.device),example_targets)[0]



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


def plot_interpolation_m1(model,z_in=None,image_n=0,latent_dim=8,imsize=28,interpolate_dim=20,std_=3,batch_size=100):
  if z_in is None:
      ztest =model.sample(batch_size=batch_size,z_out=True)
  else:
    ztest=z_in
  test_data=torch.zeros(latent_dim*interpolate_dim,latent_dim)
  for i in range(test_data.shape[0]):
      test_data[i]=ztest[image_n]

  range_=np.zeros([latent_dim,interpolate_dim])
  for c,i in enumerate(ztest.std(0).cpu().detach().numpy()):
      range_[c]=np.linspace(-std_*i,std_*i,int(interpolate_dim))

  for c in range(latent_dim*interpolate_dim):
      test_data[c][int(c/interpolate_dim)]=range_[int(c/interpolate_dim)][c%interpolate_dim]
      # print(c,int(c/interpolate   _dim),"_",int(c/interpolate_dim),c%interpolate_dim,"_",range_[int(c/interpolate_dim)][c%interpolate_dim])
      # print(c,test_data[c].cpu().detach().numpy())
  smple_pic =model.decode(test_data.to(model.device))

  all_pics=np.zeros([imsize*latent_dim,imsize*interpolate_dim])
  for i in range(latent_dim):
      for j in range(interpolate_dim):
          all_pics[i*imsize:(i+1)*imsize,j*imsize:(j+1)*imsize]=smple_pic[(interpolate_dim*i)+j][0].cpu().data
  fig = plt.figure()
  fig.set_figheight(40)
  fig.set_figwidth(20)
  plt.title("Interpolations")
  plt.imshow(all_pics, interpolation='none')
  plt.show()
  return(ztest.std(0).cpu().detach().numpy())

# could be generalized to take a specific number. It is right now harded to interpolate on class 5.
def plot_interpolation_m2(model, z_in=None,image_n=0,latent_dim=8,imsize=28,interpolate_dim=20,std_=3,batch_size=100):
  if z_in is None:
      ztest =model.sample(batch_size=batch_size,z_out=True)
  else:
    ztest=z_in
  test_data=torch.zeros(latent_dim*interpolate_dim,latent_dim)
  for i in range(test_data.shape[0]):
      test_data[i]=ztest[image_n]
  range_=np.zeros([latent_dim,interpolate_dim])
  for c,i in enumerate(ztest.std(0).cpu().detach().numpy()):
      range_[c]=np.linspace(-std_*i,std_*i,int(interpolate_dim))
  for c in range(latent_dim*interpolate_dim):
      test_data[c][int(c/interpolate_dim)]=range_[int(c/interpolate_dim)][c%interpolate_dim]
      # print(c,int(c/interpolate_dim),"_",int(c/interpolate_dim),c%interpolate_dim,"_",range_[int(c/interpolate_dim)][c%interpolate_dim])
      # print(c,test_data[c].cpu().detach().numpy())
  smple_pic = model.decode(test_data.to(model.device),y=torch.ones((model.z_dim*interpolate_dim)).type(torch.int64)*5)
  all_pics=np.zeros([imsize*latent_dim,imsize*interpolate_dim])
  for i in range(latent_dim):
      for j in range(interpolate_dim):
  #         print(i,j)
          all_pics[i*imsize:(i+1)*imsize,j*imsize:(j+1)*imsize]=smple_pic[(interpolate_dim*i)+j][0].cpu().data
  fig = plt.figure()
  fig.set_figheight(40)
  fig.set_figwidth(20)
  plt.imshow(all_pics, interpolation='none')
  plt.show()
  return(ztest.std(0).cpu().detach().numpy())


def plot_loss_m1(train_loss,test_loss):
    x_epoch = np.arange(len(train_loss))
    plt.figure(dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x_epoch, train_loss, 'r', x_epoch, test_loss, 'b')
    plt.title("Loss")
    plt.legend(['Train ELBO', 'Validation ELBO'])
    plt.xlabel('Epoch'), plt.ylabel('ELBO')
    plt.show()

def plot_loss_m2(train_loss,test_loss, train_acc, test_acc ):
    x_epoch = np.arange(len(train_loss))
    plt.figure(dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x_epoch, train_loss, 'r', x_epoch, test_loss, 'b')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title("Loss")

    x_epoch = np.arange(len(train_acc))
    plt.figure(dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x_epoch, train_acc, 'r', x_epoch, test_acc, 'b')
    plt.legend(['Train Acc', 'Validation Acc'])
    plt.xlabel('Epoch'), plt.ylabel('Acc')
    plt.title("Accuracy")
    plt.show()


def make_cm(model, dset):
  idx_test_subset = get_target_indexes(dset, classes, 10)
  X_test = dset.data[idx_test_subset].type(torch.float)
  y_test = dset.targets[idx_test_subset].numpy()

  model.eval()
  with torch.no_grad():
      y_pred = model.classifier(X_test.to(model.device)).cpu().numpy()
      y_pred = np.argmax(y_pred, axis=1)

  cm = sklearn.metrics.confusion_matrix(y_true = y_test, y_pred = y_pred)
  df_cm = pd.DataFrame(cm, index = [str(i) for i in classes],
                        columns = [str(i) for i in classes])
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True)
  plt.title("Confusion Matrix")
  plt.show()
