import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fit_m1(model, train_loader,optimizer):
    model.train()
    running_loss = 0.0
    # Run each batch in training dataset
    for idx, (images, labels) in enumerate(train_loader):
        mu, logvar, z, recon_images = model(images.to(model.device))

        loss, bce, kld = model.elbo(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

    loss_mean = running_loss/len(train_loader.dataset)
    return loss_mean

def test_m1(model, test_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            mu, logvar, z, recon_images = model(images.to(model.device))
            loss, bce, kld = model.elbo(recon_images, images, mu, logvar)
            running_loss += loss

    return running_loss/len(test_loader.dataset)



def fit_m2(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    # Run each batch in training dataset
    for idx, (images, labels) in enumerate(train_loader):
        # split in labelled / unlabelled
        x_u, x_l = train_test_split(images, test_size=0.1, shuffle=True, random_state=42)
        x_u, x_l = x_u.to(model.device), x_l.to(model.device)
        y_u, y_l = train_test_split(labels, test_size=0.1, shuffle=True, random_state=42)
        y_u, y_l = y_u.to(model.device), y_l.to(model.device)

        out_fpass = model(x_l, x_u, y_l)

        y_pred = out_fpass["L"][4].detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_l.cpu().numpy(), y_pred)
        loss = model.J_alpha(x_l, y_l, x_u, out_fpass['L'], out_fpass['U'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss
        running_acc += acc

    loss_mean = running_loss/len(train_loader)
    acc_mean = running_acc/len(train_loader)
    return loss_mean, acc_mean


def test_m2(model, test_loader):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            # split in labelled / unlabelled
            x_u, x_l = train_test_split(images, test_size=0.1, shuffle=True, random_state=42)
            x_u, x_l = x_u.to(model.device), x_l.to(model.device)
            y_u, y_l = train_test_split(labels, test_size=0.1, shuffle=True, random_state=42)
            y_u, y_l = y_u.to(model.device), y_l.to(model.device)

            out_fpass = model(x_l, x_u, y_l)

            y_pred = out_fpass["L"][4].detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            acc = accuracy_score(y_l.cpu().numpy(), y_pred)

            loss = model.J_alpha(x_l, y_l, x_u, out_fpass['L'], out_fpass['U'])

            running_loss += loss
            running_acc += acc

    loss_mean = running_loss/len(test_loader)
    acc_mean = running_acc/len(test_loader)
    return loss_mean, acc_mean



def run_epochs(mode, model, optimizer ,epochs, train_loader, test_loader, train_loss, test_loss, train_acc=None, test_acc=None):
    if mode == "vae":
        print("TODO")
    elif mode == "m1":
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")

            print("\t Training ...")
            train_epoch_loss = -1*fit_m1(model, train_loader, optimizer)

            print("\t Testing ...")
            test_epoch_loss = -1*test_m1(model, test_loader)
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)

            print(f"Train ELBO: {train_epoch_loss:.4f}")
            print(f"Val ELBO: {test_epoch_loss:.4f}")
        return (train_loss, test_loss)
    elif mode == "m2":
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")

            print("\t Training ...")
            train_epoch_loss, train_epoch_acc = fit_m2(model, train_loader, optimizer)

            print("\t Testing ...")
            test_epoch_loss, test_epoch_acc = test_m2(model, test_loader)
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            test_loss.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {test_epoch_loss:.4f}")
            print(f"Train Acc: {train_epoch_acc:.4f}")
            print(f"Val Acc: {test_epoch_acc:.4f}")

        return (train_loss, test_loss, train_acc, test_acc)
