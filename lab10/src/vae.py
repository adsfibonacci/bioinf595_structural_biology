import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import chemprop

class encoder(nn.Module):
    def __init__(self, enc_dim, h_dim, l_dim, ckpt_path):
        super().__init__()

        self.encoder = chemprop.models.MPNN.load_from_checkpoint(ckpt_path)

        self.fc1 = nn.Linear(enc_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, l_dim)
        self.fc_logvar = nn.Linear(h_dim, l_dim)

        return

    def encode(self, batch):
        hidden = self.encoder.encoding(batch.bmg, batch.V_d, batch.X_d, i=-1)
        return hidden

    def forward(self, batch):
        h = self.encode(batch)
        h = F.relu(self.fc1(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    pass

class decoder(nn.Module):
    def __init__(self, l_dim, h_dim, fp_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(l_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, fp_dim),
            nn.Sigmoid()
        )

        return

    def forward(self, z):
        return self.model(z)

    pass

class VAE(L.LightningModule):
    def __init__(self, enc_dim, h_dim, l_dim, fp_dim, ckpt_path, lr, lambda_triplet=1.0):
        super().__init__()

        self.encoder = encoder(enc_dim, h_dim, l_dim, ckpt_path)
        self.decoder = decoder(l_dim, h_dim, fp_dim)

        self.lr = lr

        self.triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.lambda_triplet = lambda_triplet

        return

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        mu, logvar = self.encoder(batch)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, z, mu, logvar

    def vae_loss(self, recon, target, mu, logvar):
        reduction = "mean"
        if target.max() <= 1.0:
            recon_loss = F.binary_cross_entropy(recon, target, reduction=reduction)
        else:
            recon_loss = F.mse_loss(recon, target, reduction=reduction)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl, recon_loss, kl

    def training_step(self, batch, batch_idx):
        recon, z, mu, logvar = self(batch["anchor"])
        target = batch["anchor"].Y

        vae_loss, recon_loss, kl = self.vae_loss(recon, target, mu, logvar)

        mu_a, _ = self.encoder(batch["anchor"])
        mu_p, _ = self.encoder(batch["positive"])
        mu_n, _ = self.encoder(batch["negative"])

        triplet_loss = self.triplet_loss_fn(mu_a, mu_p, mu_n)

        loss = vae_loss + self.lambda_triplet * triplet_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_vae": vae_loss,
                "train_recon": recon_loss,
                "train_kl": kl,
                "train_triplet": triplet_loss
            },
            on_step=True,
            on_epoch=True,
            batch_size=target.size(0)
        )

        return loss

    def validation_step(self, batch, batch_idx):
        recon, z, mu, logvar = self(batch["anchor"])
        target = batch["anchor"].Y

        vae_loss, recon_loss, kl = self.vae_loss(recon, target, mu, logvar)

        mu_a, _ = self.encoder(batch["anchor"])
        mu_p, _ = self.encoder(batch["positive"])
        mu_n, _ = self.encoder(batch["negative"])

        triplet_loss = self.triplet_loss_fn(mu_a, mu_p, mu_n)

        loss = vae_loss + self.lambda_triplet * triplet_loss

        self.log_dict(
            {
                "val_loss": loss,
                "val_vae": vae_loss,
                "val_recon": recon_loss,
                "val_kl": kl,
                "val_triplet": triplet_loss
            },
            on_epoch=True,
            batch_size=target.size(0)
        )

        return

    def test_step(self, batch, batch_idx):
        recon, z, mu, logvar = self(batch["anchor"])
        target = batch["anchor"].Y

        vae_loss, recon_loss, kl = self.vae_loss(recon, target, mu, logvar)

        mu_a, _ = self.encoder(batch["anchor"])
        mu_p, _ = self.encoder(batch["positive"])
        mu_n, _ = self.encoder(batch["negative"])

        triplet_loss = self.triplet_loss_fn(mu_a, mu_p, mu_n)

        loss = vae_loss + self.lambda_triplet * triplet_loss

        self.log_dict(
            {
                "test_loss": loss,
                "test_vae": vae_loss,
                "test_recon": recon_loss,
                "test_kl": kl,
                "test_triplet": triplet_loss
            },
            on_epoch=True,
            batch_size=target.size(0)
        )

        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer

    pass
