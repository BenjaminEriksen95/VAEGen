import torch

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

def test_m1(model, test_loader,optimizer):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            mu, logvar, z, recon_images = model(images.to(model.device))
            loss, bce, kld = model.elbo(recon_images, images, mu, logvar)
            running_loss += loss

    return running_loss/len(test_loader.dataset)


def run_epochs(mode, model, optimizer ,epochs, train_loader, test_loader, train_loss, test_loss):
    if mode == "vae":
        print("TODO")
    elif mode == "m1":
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")

            print("\t Training ...")
            train_epoch_loss = -1*fit_m1(model, train_loader, optimizer)

            print("\t Testing ...")
            test_epoch_loss = -1*test_m1(model, test_loader, optimizer)
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)

            print(f"Train ELBO: {train_epoch_loss:.4f}")
            print(f"Val ELBO: {test_epoch_loss:.4f}")
        return (train_loss, test_loss)
    elif mode == "m2":
        print("TODO")
