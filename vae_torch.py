from vrae.utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
from cmath import inf
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
features = 8
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

        # decoding
        return reconstruction, mu, log_var



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
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model):
    model.train()
    running_loss = 0.0
    # For each batch
    for batch in tqdm(range(0,len(data_load))):
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data_load[batch])
        # print(reconstruction.shape)
        # print(X_train.shape)
        # bce_loss = criterion(reconstruction, X_train)
        bce_loss = recon_loss(reconstruction,data_load[batch])
        loss = final_loss(bce_loss, mu, logvar)
        # loss = loss + 0.5* recon_loss(2*reconstruction,reconstruction)
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
            reconstruction, mu, logvar = model(data_load[batch])
            # bce_loss = criterion(reconstruction, X_train)
            bce_loss = recon_loss(reconstruction,data_load[batch])
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            full_recon_loss += bce_loss.item()

    val_loss = running_loss/len(X_train)
    full_recon_loss = full_recon_loss/len(X_train)
    print(f"Recon Loss: {full_recon_loss:.4f}")
    return val_loss, full_recon_loss

train_loss = []
val_loss = []
best_val_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model)
    val_epoch_loss, full_recon_loss = validate(model)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    #Save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(),"./vae_torch.pt")
        print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# Get Results
# Plot results
model = LinearVAE().to(device)
model.load_state_dict(torch.load("./vae_torch.pt"))
model.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model(X_train)
    test_recon_loss = recon_loss(reconstruction,X_train)
    print("Test Recon Loss: " + str(recon_loss(reconstruction,X_train)))

# Plot results
X_train_np = X_train.cpu().detach().numpy()
reconstruction = reconstruction.cpu().detach().numpy()
plt.plot(X_train_np[0,0,0,:])
# plt.show()
plt.plot(reconstruction[0,0,0,:])
# plt.show()

mu = mu.cpu().detach().numpy()

## Latent Space Extracted Features
plt.plot(mu)
# plt.show()

print(mu.shape)
# plot_clustering(mu, y_test, engine='matplotlib', download = False)

def LDA(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA(mu,y_test)
print("LDA Score: " + str(score))


f = open("vae_output.txt", "a")

f.write(f"{score}\n")

f.close()

f = open("vae_output_recon.txt", "a")

f.write(f"{test_recon_loss}\n")

f.close()

## NLL
with torch.no_grad():
    reconstruction, mu, logvar = model(X_train)
    recon_1 = recon_loss(reconstruction,X_train)
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    nll_loss = recon_1/len(X_train) + KLD_1
    print(nll_loss)

f = open("vae_output_NLL.txt", "a")

f.write(f"{nll_loss}\n")

f.close()