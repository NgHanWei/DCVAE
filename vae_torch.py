from cgi import test
from vae.utils import *
from vae_models import vanilla_vae
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
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
parser.add_argument('-features', type=int, default= 16,
                    help='Number of Latent Features', required=False)
parser.add_argument('-lr', type=float, default= 0.0005,
                    help='Set Learning Rate', required=False)
parser.add_argument('-clip', type=float, default= 0,
                    help='Set Gradient Clipping Threshold', required=False)
parser.add_argument('-data', type=str, default= 'eeg',
                    help='Choose Type of Data: eeg or semg', required=False)
parser.add_argument('-datapath', type=str, help='Path to data',required=True)
parser.add_argument('-all', default=False, action='store_true')
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
if args.data == 'eeg':
    datapath = args.datapath + '/KU_mi_smt.h5'
else:
    datapath = args.datapath + '/semg_flexex_smt.h5'

dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(0)
set_random_seeds(seed=20200205, cuda=True)

if args.data == 'eeg':

    X_train_all , y_train_all = get_multi_data([targ_subj])
    X_train_all = np.expand_dims(X_train_all,axis=1)

    if args.all == False:
        X_test , y_test = get_multi_data([targ_subj])
        X_test = np.expand_dims(X_test,axis=1)
        X_valid = X_train_all[320:360]
        X_test = X_train_all[360:]
        X_train = X_train_all[:320]
        y_train = y_test[:320]
        y_test = y_test[360:]

    else:
        # Data visualisation
        X_train = X_train_all
        X_test = X_train_all
        X_valid = X_train_all
        y_train = y_train_all
        y_test = y_train_all
## sEMG data
else:
    subject_list = list(range(1,41))
    subject_list.remove(targ_subj)
    if args.all == False:
        X_train, y_train = get_multi_data(subject_list[3:])
        X_valid, y_valid = get_multi_data(subject_list[:3])
        X_test, y_test = get_multi_data([targ_subj])
        X_train = np.expand_dims(X_train,axis=1)
        X_valid = np.expand_dims(X_valid,axis=1)
        X_test = np.expand_dims(X_test,axis=1)
    else:
        X_train, y_train = get_multi_data(subject_list)
        X_valid, y_valid = get_multi_data(subject_list)
        X_test, y_test = get_multi_data(subject_list)
        X_train = np.expand_dims(X_train,axis=1)
        X_valid = np.expand_dims(X_valid,axis=1)
        X_test = np.expand_dims(X_test,axis=1)

X_train = torch.from_numpy(X_train)
X_train = X_train.to('cuda')
X_valid = torch.from_numpy(X_valid)
X_valid = X_valid.to('cuda')
X_test = torch.from_numpy(X_test)
X_test = X_test.to('cuda')


# VAE model
input_shape=(X_train.shape[1:])
batch_size = 16
kernel_size = 5
filters = 8
features = args.features

data_load = torch.split(X_train,batch_size)
channels = len(X_train[0,0,:,0])

print("Number of Features: " + str(features))
if args.clip > 0:
    print("Gradient Clipping: " + str(args.clip))
else:
    print("No Gradient Clipping")

if args.data == 'eeg':
    print("Data Loaded: " + args.data)
else:
    print("Data Loaded: semg")


# learning parameters
epochs = args.epochs
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Define Model
model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=args.data,data_length=len(data_load[0][:,0,0,0])).to(device)
# print(model)
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)
criterion = nn.BCELoss(reduction='sum')

## Number of trainable params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable params: " + str(pytorch_total_params))

# Save file name
file_name = "./vae_torch" +  '_' + str(args.data)  + '_' + str(targ_subj) + '_' + str(filters) + '_' + str(channels) + '_' + str(features) + ".pt"

def recon_loss(outputs,targets):

    outputs = torch.flatten(outputs)
    targets = torch.flatten(targets)

    loss = nn.MSELoss()

    recon_loss = loss(outputs,targets)

    return recon_loss

def final_loss(bce_loss, mu, logvar):
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
        bce_loss = recon_loss(reconstruction,data_load[batch])
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    train_loss = running_loss/len(X_train)

    return train_loss

def validate(model):
    model.eval()
    running_loss = 0.0
    full_recon_loss = 0.0
    with torch.no_grad():
        # For each image in batch
        # for batch in range(0,len(X_valid)):
        reconstruction, mu, logvar = model(X_valid)
        bce_loss = recon_loss(reconstruction,X_valid)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        full_recon_loss += bce_loss.item()

    val_loss = running_loss/len(X_valid)
    full_recon_loss = full_recon_loss/len(X_valid)
    print(f"Recon Loss: {full_recon_loss:.4f}")
    return val_loss, full_recon_loss

def eval(model):
    model.eval()
    nll_loss_eval = 0
    with torch.no_grad():
        reconstruction, mu, logvar = model(X_test)
        recon_1 = recon_loss(reconstruction,X_test)
        KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        nll_loss_eval = (recon_1 + KLD_1)/len(X_test)
        return nll_loss_eval.item()

train_loss = []
val_loss = []
eval_loss = []
best_val_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model)
    val_epoch_loss, full_recon_loss = validate(model)
    eval_epoch_loss = eval(model)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    eval_loss.append(eval_epoch_loss)

    #Save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(),"./vae_torch.pt")
        print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# Save Training Info
df = pd.DataFrame(np.asarray(train_loss))
## save to xlsx file
filepath = 'vae_train.xlsx'
df.to_excel(filepath, index=False)
df = pd.DataFrame(np.asarray(eval_loss))
## save to xlsx file
filepath = 'vae_eval.xlsx'
df.to_excel(filepath, index=False)

model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=args.data,data_length=len(data_load[0][:,0,0,0])).to(device)
model.load_state_dict(torch.load("./vae_torch.pt"))


# open('vae_output_LDA.txt', 'w').close()
# open('vae_output_LDA2.txt', 'w').close()
# open('vae_output_recon.txt', 'w').close()
# open('vae_output_NLL.txt', 'w').close()

## Anomaly Detection
model.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model(X_test)
    reconstruction_train, mu_train, logvar_train = model(X_train)
    test_recon_loss = recon_loss(reconstruction,X_test)

if (test_recon_loss > 20000 or torch.isnan(test_recon_loss)) and args.data == 'eeg':
    # print("Anomaly Detected")
    index_list = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        data_test = torch.split(X_test,1)
        for i in range(0,len(data_test)):
            reconstruction, mu, logvar = model(data_test[i])
            recon_1 = recon_loss(reconstruction,data_test[i])
            KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            nll_loss = (recon_1 + KLD_1)
            if recon_1 > 50000 or torch.isnan(test_recon_loss):
                index_list.append(i)
            total_loss += nll_loss

        if len(index_list) == 40:
            index_list = []
            for i in range(0,len(data_test)):
                reconstruction, mu, logvar = model(data_test[i])
                recon_1 = recon_loss(reconstruction,data_test[i])
                if recon_1 > 100000 or torch.isnan(recon_1):
                    index_list.append(i)

    old_list = list(range(0,40))
    new_list = [number for number in old_list if number not in index_list]
    X_test = X_test[new_list,:,:,:]
    y_test = y_test[new_list]


# Get Results
# Plot results

model.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model(X_test)
    reconstruction_train, mu_train, logvar_train = model(X_train)
    test_recon_loss = recon_loss(reconstruction,X_test)
    print("Test Recon Loss: " + str(recon_loss(reconstruction,X_test)))

# Plot results
X_train_np = X_test.cpu().detach().numpy()
reconstruction = reconstruction.cpu().detach().numpy()
# print(X_train_np.shape)
plt.plot(X_train_np[0,0,0,:])
# plt.show()
plt.plot(reconstruction[0,0,0,:])
# plt.show()

mu = mu.cpu().detach().numpy()
mu_train = mu_train.cpu().detach().numpy()
## Latent Space Extracted Features
plt.plot(mu)
# plt.show()

# fig, axs = plt.subplots(features, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
# fig.suptitle('Learned Latent Features')
# hex_colors = []
# for _ in range(0,features):
#     hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
# colors = [hex_colors[int(i)] for i in range(0,features)]
# for i in range(0,features):
#     axs[i].plot(mu[:,i], linewidth=3, color=colors[i])

# Hide x labels and tick labels for all but bottom plot.
# for ax in axs:
#     ax.label_outer()

# plt.show()

# print(mu.shape)
plot_clustering(mu, y_test, engine='matplotlib', download = False)

def LDA(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(mu_train,y_train)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA(mu,y_test)
print("LDA Score: " + str(score))

f = open("vae_output_LDA_train.txt", "a")
f.write(f"{score}\n")
f.close()

def LDA_2(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA_2(mu,y_test)

f = open("vae_output_LDA_test.txt", "a")
f.write(f"{score}\n")
f.close()

f = open("vae_output_recon.txt", "a")
f.write(f"{test_recon_loss}\n")
f.close()

## NLL
with torch.no_grad():
    reconstruction, mu, logvar = model(X_test)
    recon_1 = recon_loss(reconstruction,X_test)
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    nll_loss = (recon_1 + KLD_1)/len(X_test)
    print(nll_loss)

f = open("vae_output_NLL.txt", "a")
f.write(f"{nll_loss}\n")
f.close()
