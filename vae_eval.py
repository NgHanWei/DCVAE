from cmath import inf
from vrae.utils import *
from vae_models import dualchain_vae
from vae_models import vanilla_vae
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import h5py
import pandas as pd
import sys
import torch
import torch.nn as nn
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

parser = argparse.ArgumentParser(
    description='Model Evaluation')
parser.add_argument('-all', default=False, action='store_true')
parser.add_argument('-pretrained', default=False, action='store_true')
parser.add_argument('-datapath', type=str, help='Path to training data',required=True)
parser.add_argument('-model', type=str, help='Path to model data',required=True)
parser.add_argument('-alpha', type=float, default= 0.5,
                    help='Alpha Hyperparameter for First Chain KL Divergence', required=False)
parser.add_argument('-beta', type=float, default= 0.5,
                    help='Beta Hyperparameter for Second Chain KL Divergence', required=False)
args = parser.parse_args()

string = args.model

x = string.split("_")
index = x.index('torch')
x = x[index + 1:]

data = x[0]
targ_subj = int(x[1])
filters = int(x[2])
channels = int(x[3])
features = int(x[4][:2])

if args.pretrained == True:
    data = 'eeg'
    targ_subj = 6
    filters = 8
    channels = 62
    features = 16

if data == 'eeg':
    datapath = args.datapath + '/KU_mi_smt.h5'
else:
    datapath = args.datapath + '/semg_flexex_smt.h5'
dfile = h5py.File(datapath, 'r')

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

if data == 'eeg':

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

# print(X_train.shape)
X_train = torch.from_numpy(X_train)
X_train = X_train.to('cuda')
X_valid = torch.from_numpy(X_valid)
X_valid = X_valid.to('cuda')
X_test = torch.from_numpy(X_test)
X_test = X_test.to('cuda')
data_load = torch.split(X_train,16)

def recon_loss(outputs,targets):

    outputs = torch.flatten(outputs)
    targets = torch.flatten(targets)

    loss = nn.MSELoss()

    recon_loss = loss(outputs,targets)

    return recon_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'dual' in args.model:
    features = int(features / 2)
    model = dualchain_vae.DCVAE(filters=filters,channels=channels,features=features,data_type=data).to(device)
else:
    features = features
    model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(data_load[0][:,0,0,0])).to(device)

# Plot results
if args.pretrained == False:
    if 'dual' in args.model:
        model.load_state_dict(torch.load(string))
    else:
        model.load_state_dict(torch.load(string))
else:
    if 'dual' in args.model:
        model.load_state_dict(torch.load("./pre_trained/dual_vae_torch.pt"))
    else:
        model.load_state_dict(torch.load("./pre_trained/vae_torch.pt"))
# open('dual_output_LDA.txt', 'w').close()
# open('dual_output_LDA2.txt', 'w').close()
# open('dual_output_recon.txt', 'w').close()
# open('dual_output_NLL.txt', 'w').close()

## Anomaly detection
model.eval()
with torch.no_grad():
    if 'dual' in args.model:
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_test)
        reconstruction_train, mu_train, logvar_train, new_inputs_train, reconstruction_2_train,mu_2_train,logvar_2_train = model(X_train)
        recon_full = reconstruction + reconstruction_2
        # print(recon_loss(reconstruction,X_train))
        test_recon_loss = recon_loss(reconstruction+reconstruction_2,X_test)
    else:
        reconstruction, mu, logvar = model(X_test)
        reconstruction_train, mu_train, logvar_train = model(X_train)
        test_recon_loss = recon_loss(reconstruction,X_test)

if (test_recon_loss > 20000 or torch.isnan(test_recon_loss)) and data == 'eeg':

    index_list = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        data_test = torch.split(X_test,1)
        for i in range(0,len(data_test)):
            reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_test[i])
            recon_1 = recon_loss(reconstruction_2,new_inputs)
            # print(recon_1)
            if recon_1 > 50000 or torch.isnan(recon_1):
                index_list.append(i)
        if len(index_list) == 40:
            index_list = []
            for i in range(0,len(data_test)):
                reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_test[i])
                recon_1 = recon_loss(reconstruction_2,new_inputs)
                recon_2 = recon_loss(reconstruction+reconstruction_2,data_test[i])
                if recon_1 > 100000 or torch.isnan(recon_1):
                    index_list.append(i)

    old_list = list(range(0,40))
    new_list = [number for number in old_list if number not in index_list]
    X_test = X_test[new_list,:,:,:]
    y_test = y_test[new_list]

## Evaluation
model.eval()
with torch.no_grad():
    if 'dual' in args.model:
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_test)
        reconstruction_train, mu_train, logvar_train, new_inputs_train, reconstruction_2_train,mu_2_train,logvar_2_train = model(X_train)
        recon_full = reconstruction + reconstruction_2
        # print(recon_loss(reconstruction,X_train))
        test_recon_loss = recon_loss(reconstruction+reconstruction_2,X_test)
        full_recon = torch.flatten(reconstruction+reconstruction_2) - torch.flatten(X_test)
        print("Test Recon Loss: " + str(recon_loss(reconstruction+reconstruction_2,X_test)))
    else:
        reconstruction, mu, logvar = model(X_test)
        reconstruction_train, mu_train, logvar_train = model(X_train)
        test_recon_loss = recon_loss(reconstruction,X_test)

# Original
X_train_np = X_test.cpu().detach().numpy()
plt.plot(X_train_np[0,0,0,:])
plt.show()
# Recon 1
reconstruction_np = reconstruction.cpu().detach().numpy()
plt.plot(reconstruction_np[0,0,0,:])
plt.show()
# Recon 2
if 'dual' in args.model:
    recon_full = recon_full.cpu().detach().numpy()
    plt.plot(recon_full[0,0,0,:])
    plt.show()

    mu = mu.cpu().detach().numpy()
    mu_2 = mu_2.cpu().detach().numpy()
    mu_3 = np.concatenate((mu,mu_2),axis=1)

    mu_train = mu_train.cpu().detach().numpy()
    mu_2_train = mu_2_train.cpu().detach().numpy()
    mu_3_train = np.concatenate((mu_train,mu_2_train),axis=1)
else:
    mu_3 = mu.cpu().detach().numpy()
    mu_3_train = mu_train.cpu().detach().numpy()

plt.plot(mu_3)
plt.show()


fig, axs = plt.subplots(features, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle('Learned Latent arg.features')
hex_colors = []
for _ in range(0,features):
    hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
colors = [hex_colors[int(i)] for i in range(0,features)]
for i in range(0,features):
    axs[i].plot(mu_3[:,i], linewidth=3, color=colors[i])

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

# plt.show()


## PCA and TSNE

plot_clustering(mu_3, y_test, engine='matplotlib', download = False)

## LDA

def LDA_1(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(mu_3_train,y_train)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA_1(mu_3,y_test)
print("LDA Score: " + str(score))

f = open("dual_output_LDA.txt", "a")
f.write(f"{score}\n")
f.close()

def LDA_2(x,y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y)
    lda_score = lda.score(x,y)
    return lda_score

score = LDA_2(mu_3,y_test)

f = open("dual_output_LDA2.txt", "a")
f.write(f"{score}\n")
f.close()

f = open("dual_output_recon.txt", "a")
f.write(f"{test_recon_loss}\n")
f.close()

## NLL
model.eval()
with torch.no_grad():
    
    if 'dual' in args.model:
    
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_test)
        recon_1 = recon_loss(reconstruction_2,new_inputs)
        KLD_1 = args.alpha * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD_2 = args.beta * -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
        nll_loss = (recon_1 + KLD_2 + KLD_1)/len(X_test)
    else:
        reconstruction, mu, logvar = model(X_test)
        recon_1 = recon_loss(reconstruction,X_test)
        KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        nll_loss = (recon_1 + KLD_1)/len(X_test)
    
    print(nll_loss)


f = open("dual_output_NLL.txt", "a")
f.write(f"{nll_loss}\n")
f.close()
