from cmath import inf
import math
from vrae.utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.distributions import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

from braindecode.torch_ext.util import set_random_seeds
import h5py
set_random_seeds(seed=0, cuda=True)

parser = argparse.ArgumentParser(
    description='VAE Subject Selection')
parser.add_argument('-subj', type=int,
                    help='Target Subject for Subject Selection', required=True)
parser.add_argument('-epochs', type=int, default= 100,
                    help='Number of Epochs', required=False)
args = parser.parse_args()

targ_subj = args.subj

# Get data from single subject.
def get_data(subj):
    dpath = 's' + str(subj)
    X = dfile[dpath]['X']
    Y = dfile[dpath]['Y']
    return X, Y

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

# Randomly shuffled subject.
datapath = 'D:/DeepConvNet/pre-processed/KU_mi_smt.h5'
fold = 0
subjs = [35, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]
test_subj = subjs[fold]
cv_set = np.array(subjs[fold+1:] + subjs[:fold])

dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(0)
set_random_seeds(seed=20200205, cuda=True)
            
independent = np.arange(51, 55, 1, dtype=int)

X_train , y_train = get_multi_data([targ_subj])
X_test , y_test = get_multi_data([targ_subj])
X_train = np.expand_dims(X_train,axis=1)
X_test = np.expand_dims(X_test,axis=1)

sub_labels = np.zeros(200)
sub_labels = np.concatenate((sub_labels,np.ones(200)),axis=0)
# print(sub_labels.shape)

print(X_train.shape)
X_train = torch.from_numpy(X_train)
X_train = X_train.to('cuda')

# VAE model
input_shape=(X_train.shape[1:])
batch_size = 16
kernel_size = 5
filters = 8
features = 4
data_load = torch.split(X_train,batch_size)

# leanring parameters
epochs = args.epochs
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=filters*2,out_channels=filters*4, kernel_size=(62,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.flatten =  nn.Flatten()
        self.dense = nn.Linear(156*filters,32)
        self.dense2 = nn.Linear(features,32)
        self.dense3 = nn.Linear(32,156*filters)
        self.dense_features = nn.Linear(32,features)
        # decoder

        self.decode_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*4, kernel_size=(62,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
        )

        self.decode_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
        )

        self.output = nn.ConvTranspose2d(in_channels=filters*2,out_channels=1,kernel_size=5,padding=(2,2))

        ## Second Encoder/Decoder

        self.layer1_2 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(in_channels=filters*2,out_channels=filters*4, kernel_size=(62,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.flatten_2 =  nn.Flatten()
        self.dense_2 = nn.Linear(156*filters,32)
        self.dense2_2 = nn.Linear(features,32)
        self.dense3_2 = nn.Linear(32,156*filters)
        self.dense_features_2 = nn.Linear(32,features)
        # decoder

        self.decode_layer1_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*4, kernel_size=(62,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
        )

        self.decode_layer2_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
        )

        self.output_2 = nn.ConvTranspose2d(in_channels=filters*2,out_channels=1,kernel_size=5,padding=(2,2))

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        # print(x.shape)
        original = x
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.dense(x))
        # print(x.shape)

        # get `mu` and `log_var`
        mu = self.dense_features(x)
        log_var = self.dense_features(x) + 1e-8
        # print(mu.shape)
        # print(log_var.shape)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print(z.shape)

        # decoding
        x = F.relu(self.dense2(z))
        x = F.relu(self.dense3(x))
        # print(x.shape)
        x = x.view(-1, filters*4, 1, 39)
        # print(x.shape)

        x = self.decode_layer1(x)
        # print(x.shape)
        x = self.decode_layer2(x)
        # print(x.shape)
        reconstruction = self.output(x)
        # print(reconstruction.shape)

        new_inputs = original - reconstruction
        ## Second encoder decoder
        # print(new_inputs.shape)
        x1 = self.layer1_2(new_inputs)
        x1 = self.layer2_2(x1)
        x1 = self.flatten_2(x1)
        x1 = F.relu(self.dense_2(x1))

        # get `mu` and `log_var`
        mu_2 = self.dense_features_2(x1)
        log_var_2 = self.dense_features_2(x1) + 1e-8
        # get the latent vector through reparameterization
        z_2 = self.reparameterize(mu_2, log_var_2)

        # decoding
        x1 = F.relu(self.dense2_2(z_2))
        x1 = F.relu(self.dense3_2(x1))
        # print(x1.shape)
        x1 = x1.view(-1, filters*4, 1, 39)

        x1 = self.decode_layer1_2(x1)
        x1 = self.decode_layer2_2(x1)
        reconstruction_2 = self.output_2(x1)

        # decoding
        return reconstruction, mu, log_var, new_inputs, reconstruction_2, mu_2, log_var_2



net = LinearVAE()
print(net)



## Model
model = LinearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)
criterion = nn.BCELoss(reduction='sum')

## Number of trainable params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable params: " + str(pytorch_total_params))

def recon_loss(outputs,targets):

    outputs = torch.flatten(outputs)
    targets = torch.flatten(targets)

    loss = nn.MSELoss()

    recon_loss = loss(outputs,targets)

    return recon_loss

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.25 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD
    # return BCE + KLD

def fit(model):
    model.train()
    running_loss = 0.0
    # For each batch
    for batch in tqdm(range(0,len(data_load))):
        optimizer.zero_grad()
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_load[batch])
        # print(reconstruction.shape)
        # print(X_train.shape)
        # bce_loss = criterion(reconstruction, X_train)
        bce_loss = recon_loss(reconstruction,data_load[batch])
        loss = final_loss(bce_loss, mu, logvar)
        mask_loss = recon_loss(reconstruction_2,new_inputs)
        full_loss = recon_loss(reconstruction+reconstruction_2,data_load[batch])
        mask_vae_loss = -0.25 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
        # loss = loss + mask_loss + bce_loss + mask_vae_loss
        loss = loss + mask_vae_loss + mask_loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(X_train)

    return train_loss

def validate(model):
    model.eval()
    running_loss = 0.0
    full_recon_loss = 0.0
    with torch.no_grad():
        # For each image in batch
        for batch in range(0,len(data_load)):
            reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_load[batch])
            # bce_loss = criterion(reconstruction, X_train)
            bce_loss = recon_loss(reconstruction,data_load[batch])
            loss = final_loss(bce_loss, mu, logvar)
            # loss = loss + recon_loss(2*reconstruction,reconstruction)
            mask_loss = recon_loss(reconstruction_2,new_inputs)
            full_mask_loss = recon_loss(reconstruction+reconstruction_2,data_load[batch])
            mask_vae_loss = -0.25 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
            loss = loss + mask_loss + mask_vae_loss
            running_loss += loss.item()
            full_recon_loss += full_mask_loss.item()

    val_loss = running_loss/len(X_train)
    full_recon_loss = full_recon_loss/len(X_train)
    print(f"Recon Loss: {bce_loss:.4f}")
    print(f"Full Recon Loss: {full_recon_loss:.4f}")
    return val_loss, full_recon_loss

train_loss = []
val_loss = []
recon_loss_array = []
best_val_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model)
    val_epoch_loss, full_recon_loss = validate(model)

    #Save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(),"./dual_vae_torch.pt")
        print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")    
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    recon_loss_array.append(full_recon_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

# Plot results
model = LinearVAE().to(device)
model.load_state_dict(torch.load("./dual_vae_torch.pt"))

# model.train()
# with torch.no_grad():
#     for batch in tqdm(range(0,len(data_load))):
#             reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_load[batch])

model.eval()
with torch.no_grad():
    reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_train)
    recon_full = reconstruction + reconstruction_2
    # print(recon_loss(reconstruction,X_train))
    test_recon_loss = recon_loss(reconstruction+reconstruction_2,X_train)
    full_recon = torch.flatten(reconstruction+reconstruction_2) - torch.flatten(X_train)
    print("Test Recon Loss: " + str(recon_loss(reconstruction+reconstruction_2,X_train)))

X_train_np = X_train.cpu().detach().numpy()
reconstruction_np = reconstruction.cpu().detach().numpy()
recon_full = recon_full.cpu().detach().numpy()
plt.plot(X_train_np[0,0,0,:])
# plt.show()
plt.plot(reconstruction_np[0,0,0,:])
# plt.show()
plt.plot(recon_full[0,0,0,:])
# plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.show()

plt.plot(recon_loss_array,label="recon")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.show()

mu = mu.cpu().detach().numpy()
mu_2 = mu_2.cpu().detach().numpy()
mu_3 = np.concatenate((mu,mu_2),axis=1)
print(mu_3.shape)

## Latent Space Extracted Features
plt.plot(mu)
# plt.show()
plt.plot(mu_2)
# plt.show()
plt.plot(mu_3)
# plt.show()

## PCA and TSNE

# plot_clustering(mu, y_test, engine='matplotlib', download = False)
# plot_clustering(mu_2, y_test, engine='matplotlib', download = False)
# plot_clustering(mu+mu_2, y_test, engine='matplotlib', download = False)
# plot_clustering(mu_3, y_test, engine='matplotlib', download = False)

## LDA

def LDA(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA(mu,y_test)
print(score)
score = LDA(mu_2,y_test)
print(score)
score = LDA(mu+mu_2,y_test)
print(score)
score = LDA(mu_3,y_test)
print("LDA Score: " + str(score))

f = open("dual_output.txt", "a")

f.write(f"{score}\n")

f.close()

f = open("dual_output_recon.txt", "a")

f.write(f"{test_recon_loss}\n")

f.close()

## NLL
with torch.no_grad():
    reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_train)
    recon_1 = recon_loss(reconstruction_2,new_inputs)
    KLD_1 = -0.25 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_2 = -0.25 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())

    nll_loss = recon_1/len(X_train) + KLD_2 + KLD_1
    print(nll_loss)



f = open("dual_output_NLL.txt", "a")

f.write(f"{nll_loss}\n")

f.close()
