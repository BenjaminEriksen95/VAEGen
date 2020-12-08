import torch


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(torch.nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class M1(torch.nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(M1, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            Flatten()
        )

        self.mu = torch.nn.Linear(h_dim, z_dim)
        self.logsigma = torch.nn.Linear(h_dim, z_dim)
        self.upscale_z = torch.nn.Linear(z_dim, h_dim)

        self.decoder = torch.nn.Sequential(
            UnFlatten(),
            torch.nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=0, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                32, image_channels, kernel_size=5, stride=1),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size())
        z = mu.to(device) + std.to(device) * eps.to(device)
        return z

    def encode(self, x):
        # pass x through conv encoder and flatten output to obtain h
        h = self.encoder(x)
        # use relu to prevent negative values
        mu, logvar = self.mu(h), self.logsigma(h)
        return mu, logvar

    def decode(self, z):
        h = self.upscale_z(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        # encode x to mu and logvar
        mu, logvar = self.encode(x)

        # sample z using mu and logvar
        z = self.reparameterize(mu, logvar)

        # decode z to get reconstruction
        x_recon = self.decode(z)

        return mu, logvar, z, x_recon

    def elbo(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.to(
            device), size_average=False, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # TODO: is this mean or sum?
        KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE - KLD, BCE, KLD

    def sample(self, batch_size: int = 100):
        """Sample z~p(z) and return p(x|z) / reconstruction
        """
        # sample z ~ N(0,I)
        mu, logvar = torch.zeros((batch_size, self.z_dim)), torch.ones(
            batch_size, self.z_dim)
        z = self.reparameterize(mu, logvar)

        # decode sampled z with condition
        x_recon = self.decode(z)
        return x_recon.detach()  # no gradient_fn
