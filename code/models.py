import torch
import numpy as np
from modules import Flatten, UnFlatten, UnFlatten_FF, NNprint
import torch.nn.functional as F

"""
TODO: upgrade M2 classifier: change to CNN, test regularization

"""


## Auxilliary function
def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)


## VAE
class VAE(torch.nn.Module):
    def __init__(self, device, image_channels=1, h_dim=1024, z_dim=32,num_labels=0):
        super(VAE, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = torch.nn.Sequential(
            NNprint(),
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            NNprint(),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            NNprint(),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            NNprint(),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            NNprint(),
            torch.nn.LeakyReLU(),
            Flatten(),
            NNprint(),

        )
        self.h_dim=h_dim
        self.num_labels=num_labels
        self.fc1 = torch.nn.Linear(h_dim, z_dim)
        self.fc2 = torch.nn.Linear(h_dim, z_dim)
        self.fc3 = torch.nn.Linear(z_dim+num_labels, h_dim)

        self.decoder = torch.nn.Sequential(
            NNprint(),
            UnFlatten(),
            NNprint(),
            torch.nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(),
            NNprint(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.LeakyReLU(),
            NNprint(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4,padding=0, stride=2),
            torch.nn.LeakyReLU(),
            NNprint(),
            torch.nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=1),
            torch.nn.Sigmoid(),
            NNprint(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu.to(self.device) + std.to(self.device) * esp.to(self.device)
        return z

    def bottleneck(self, h,labels):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        if self.num_labels>0:
            z=torch.cat((z,torch.nn.functional.one_hot(labels,self.num_labels)
                            .type(torch.float).to(self.device)),1)
        return z, mu, logvar

    def encode(self, x,labels):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h,labels)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x,labels)
        z = self.decode(z)
        return z, mu, logvar

    def loss_fn(self,recon_x, x, mu, logvar):
      BCE = F.binary_cross_entropy(recon_x, x.to(self.device), size_average=False,reduction='sum' )
      # see Appendix B from VAE paper:
      # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
      # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

      return BCE + KLD, BCE, KLD

    def sample(self, batch_size: int = 100,z_out=False):
        """Sample z~p(z) and return p(x|z) / reconstruction
        """
        # sample z ~ N(0,I)
        mu, logvar = torch.zeros((batch_size, self.z_dim)),torch.ones(batch_size, self.z_dim)
        z = self.reparameterize(mu, logvar)
        if z_out:
            return z
        # decode sampled z with condition
        else:
            x_recon = self.decode(z)
            return x_recon.detach()  # no gradient_fn

## M1
class M1(torch.nn.Module):
    def __init__(self, device, image_channels=1, h_dim=1024, z_dim=32):
        super(M1, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.LeakyReLU(),
        )

        self.mu = torch.nn.Linear(h_dim, z_dim) # mean
        self.logsigma = torch.nn.Linear(h_dim, z_dim) # standard deviation
        self.upscale_z = torch.nn.Linear(z_dim, h_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.LeakyReLU(),
            UnFlatten(),
            torch.nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=0, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                32, image_channels, kernel_size=5, stride=1),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.exp()
        eps = torch.randn(*mu.size())
        z = mu.to(self.device) + std.to(self.device) * eps.to(self.device)
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
            self.device), size_average=False, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # TODO: is this mean or sum?
        KLD = 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        #KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE - KLD, BCE, KLD

    def sample(self, batch_size: int = 100,z_out=False):
        """Sample z~p(z) and return p(x|z) / reconstruction
        """
        # sample z ~ N(0,I)
        mu, logvar = torch.zeros((batch_size, self.z_dim)), torch.ones(
            batch_size, self.z_dim)
        z = self.reparameterize(mu, logvar)
        if z_out:
            return z
        # decode sampled z with condition
        else:
            x_recon = self.decode(z)
            return x_recon.detach()  # no gradient_fn

## M2
class M2(torch.nn.Module):
    def __init__(self, device, image_channels=1, h_dim=1024, z_dim=32, num_labels=0):
        super(M2, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_labels=num_labels

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.LeakyReLU()
        )

        self.mu = torch.nn.Linear(h_dim + num_labels, z_dim) # mean
        self.logsigma = torch.nn.Linear(h_dim + num_labels, z_dim) # standard deviation
        self.upscale_z = torch.nn.Linear(z_dim+num_labels, h_dim)


        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.LeakyReLU(),
            UnFlatten(),
            torch.nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=.2),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=0, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                32, image_channels, kernel_size=5, stride=1),
            torch.nn.Sigmoid()
        )

        # simple FFNN classifier
        # input: flattened vector, output: probability of each class
        # TODO: implement CNN classifier
        self.classifier = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_labels), # 3 output layer
            torch.nn.Softmax(dim=1)
        )

    def reparameterize(self, mu, logvar):
        std = logvar.exp()
        eps = torch.randn(*mu.size())
        z = mu.to(self.device) + std.to(self.device) * eps.to(self.device)
        return z

    def encode(self, x, y):
        # pass x through conv encoder and flatten output to obtain h
        # n.b. normally [x,y] would be encoded, but this is not possible with a
        # CNN architecture
        h = self.encoder(x)

        # concat y (labels) to h, to place in latent space
        h = torch.cat((h, torch.nn.functional.one_hot(y, self.num_labels).type(torch.float).to(self.device)), 1)

        # fc1 encodes [h,y] into mu; fc2 encodes [h,y] into logvar
        mu, logvar = self.mu(h), self.logsigma(h) # use relu to prevent negative values

        return mu, logvar

    def decode(self, z, y):
        # concat y (labels) to z_lab, to place in latent space
        z_y = torch.cat((z, torch.nn.functional.one_hot(y, self.num_labels).type(torch.float).to(self.device)), 1)

        # upscale [z,y] to h_dim
        h = self.upscale_z(z_y)

        # decode upscaled [z,y] to x_recon
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x_labelled, x_unlabelled, y):
        # labeled case ########################################################
        # encode [x_labelled,y] to mu_lab and logvar_lab
        mu_lab, logvar_lab = self.encode(x_labelled, y)

        # sample z_lab using mu_lab and logvar_lab
        z_lab = self.reparameterize(mu_lab, logvar_lab)

        # decode [z_lab,y] to get reconstruction
        x_lab_recon = self.decode(z_lab, y)


        # classification ######################################################
        # labelled case - this is used for the classification objective loss
        y_labelled_pred = self.classifier(x_labelled)

        # unlabelled case - predict a y for the entropy term, H(q(y|x))
        y_unlabelled_pred = self.classifier(x_unlabelled)

        # unlabeled case ######################################################
        # integrate over y -- i.e. perform generation and inference for any
        # possible class of y. Then calculate loss for any case
        # We do this by tiling x_unlabelled and y, then concat for parallelism

        # TODO: repeat copies the data, so there might be a less expensive way to do this
        x_unlab_tiled = x_unlabelled.repeat(self.num_labels,1,1,1)
        # tensor of class labels [0,1,2,3,4,5,6,7,8,9], repeated for number of datapoints as [0_0,0_1,0_n, ...]
        y_unlab_tiled = torch.repeat_interleave(torch.arange(self.num_labels), x_unlabelled.shape[0])

        # encode x_unlab and y_unlab to mu_unlab and logvar_unlab
        mu_unlab, logvar_unlab = self.encode(x_unlab_tiled, y_unlab_tiled)

        # sample z_unlab via mu_unlab and logvar_unlab
        z_unlab = self.reparameterize(mu_unlab, logvar_unlab)

        # decode [z_unlab,y_unlab] to get reconstruction
        x_unlab_recon = self.decode(z_unlab, y_unlab_tiled)

        # TODO: Find better solution
        return {'L' : (mu_lab, logvar_lab, z_lab, x_lab_recon, y_labelled_pred), \
                'U' : (mu_unlab, logvar_unlab, z_unlab, x_unlab_recon, y_unlabelled_pred)}

    def log_prob_gaussian(self, x, mu, log_sigma):
        """Calculate ...
        todo
        """
        eps = 1e-10
        sigma = log_sigma.exp()
        logprob = -torch.log(sigma * np.sqrt(2*np.pi) + eps) - 0.5 * ((x - mu)/sigma)**2
        return logprob

    def Lb(self, x, x_recon, yp, z, mu, logvar):
      """Calculate lower bound, -L(x,y)

      -L(x,y) is used to calculate loss for both the Labelled and Unlabelled case

      Formula:
        -L(x,y) = E_q(z|x,y) [ logp(x|y,z) + logp(y) + logp(z) - logq(z|x,y) ]

      Returns
        Lower bound. Should return negative value.
        For loss: change sign (so loss is positive) and take mean of batch to minimize
      """
      logpx = -F.binary_cross_entropy(x_recon, x.to(self.device), size_average=False, reduce=False) # negate or not?
      logpx = reduce_sum(logpx) # sum loss per datapoint, but not over batch
      logpy = torch.log(yp) # natural log, ln
      logpz = reduce_sum(self.log_prob_gaussian(z, torch.zeros_like(z), torch.ones_like(z))) # logprob Gaussian for z, 0, I # suspect
      logqz = reduce_sum(self.log_prob_gaussian(z, mu, logvar)) # logprob Gaussian for z, mu, logvar
      lower_bound = logpx + logpy + logpz - logqz
      return lower_bound # N.B. we don't take the mean here!

    def H(self, p):
      """Calculate Entropy, H(p)
      Where
        H(p) = -∑p(i) * log(p(i))
      """
      eps = 1e-10
      return -torch.sum((p * torch.log(p + eps)), dim=1)

    def J_alpha(self, x_labelled, y_labelled, x_unlabelled, Ls, Us, alpha=.1):
      """Calculate loss as J_alpha

      J_alpha = J + alpha * C

      Where
        J = ∑L(x,y) + ∑U(x)
        -L(x,y) = E_q(z|x,y)[logp(x|y,z) + logp(y) + logp(z) - logq(z|x,y)]
        -U(x) = ∑_y q(y|x)(-L(x,y)) + H(q(y|x))
        C = E_p(x,y) [-logq(y|x)]
      """
      # labelled loss ##########################################################
      mu, logvar = Ls[0], Ls[1]
      z = Ls[2]
      x_recon = Ls[3]
      # y_prob is just a uniform dist over the labels, i.e. 1/n prob per class
      y_prob = torch.Tensor([(1/self.num_labels)]).to(self.device)

      # -L(x,y), note the sign
      Lxy = self.Lb(x_labelled, x_recon, y_prob, z, mu, logvar) # (10,), when using 10% labelled data

      # unlabelled loss ########################################################
      mu, logvar = Us[0], Us[1]
      z = Us[2]
      x_recon = Us[3]
      # y_prob is just a uniform dist over the labels, i.e. 1/n prob per class
      y_prob = torch.Tensor([(1/self.num_labels)]).to(self.device)

      # use list comprehension to get probabilities for class i for an entire batch
      # i.e. first probs for class == 0, then class == 1, etc.
      # y_pred_tiled = torch.cat([y_pred[:,i] for i in range(self.num_labels)])

      # integrate over y
      y_pred = Us[4]
      tiled_size = x_recon.shape[0]
      batch_size = y_pred.shape[0]
      n_classes = int(tiled_size/batch_size)
      U = torch.zeros(batch_size).to(self.device) # (90,), # when using 90% labelled data
      # iterate over tiles, i.e. ∑y q(y|x)(-L(x,y))
      for i in range(n_classes):
        # indexes for slicing
        l = i*y_pred.shape[0]
        r = l+y_pred.shape[0]
        # y_slice, i.e. y_pred should be shape (batch,n_classes)
        y_slice = y_pred[:,i] # probabilities for class i for whole batch (batch,) = (90,)
        # -L(x,y), note the sign
        Luxy = self.Lb(x_unlabelled, x_recon[l:r,:], y_prob, z[l:r], mu[l:r,:], logvar[l:r,:]) # (batch,) = (90,)
        U += (y_slice * Luxy) # (batch,) = (90,)

      # add entropy H to each datapoint (which is summed over y's in loop above)
      H = self.H(y_pred) # (batch,) = (90,)
      U = U + H # no mean yet, because we are still doing computing per datapoint

      # classification loss ####################################################
      y = y_labelled
      y_labelled_pred = Ls[4]
      C = F.cross_entropy(y_labelled_pred, y, reduce=False)

      # final loss #############################################################
      J = -Lxy.mean() + -U.mean()
      Ja = J + alpha * C.mean()
      return Ja


    def sample(self, y: torch.Tensor()):
        """Sample z, conditioning on y and reconstruct

        y : torch.int64
        """
        batch_size = y.size(0)

        # sample z ~ N(0,I)
        mu, logvar = torch.zeros((batch_size, self.z_dim)), torch.ones(
            batch_size, self.z_dim)

        z = self.reparameterize(mu, logvar)

        # decode sampled z with condition
        x_recon = self.decode(z, y)
        return {"xr" : x_recon.detach(), "z" : z.detach()} # no gradient_fn
