'''Preprocessor for sEMG Data.
'''
import argparse
from os.path import join as pjoin

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Preprocessor for KU Data')
parser.add_argument('--source', type=str, default='D:/sEMG_data/sEMG-dataset/filtered/mat/', help='Path to raw KU data',required=False)
parser.add_argument('--target', type=str, default='D:/eeg_vrae/pre-processed-semg', help='Path to pre-processed KU data',required=False)
args = parser.parse_args()

src = args.source
out = args.target

def get_data(subj):
    
    filename = '{:01d}_filtered.mat'.format(subj)

    filepath = pjoin(src, filename)
    raw = loadmat(filepath)
    # Obtain input: convert (time, chan) into (chan, time)
    X = np.moveaxis(raw['data'], 0,-1)
    X = decimate(X, 8)
    # Obtain Extension, (trials, chan, time)
    X1 = X[:,14*250:20*250]
    X2 = X[:,148*250:154*250]
    X3 = X[:,(134*2+14)*250:(134*2+20)*250]
    X4 = X[:,(134*3+14)*250:(134*3+20)*250]
    X5 = X[:,(134*4+14)*250:(134*4+20)*250]
    # Obtain Flexion
    X6 = X[:,24*250:30*250]
    X7 = X[:,158*250:164*250]
    X8 = X[:,(134*2+24)*250:(134*2+30)*250]
    X9 = X[:,(134*3+24)*250:(134*3+30)*250]
    X10 = X[:,(134*4+24)*250:(134*4+30)*250]
    X_new = np.array([X1, X2, X3,X4,X5,X6,X7,X8,X9,X10])
    print(X_new.shape)

    # Obtain Labels 0 - Extension, 1 - Flexion
    Y = np.array([0,0,0,0,0,1,1,1,1,1])
    print(Y.shape)

    return X_new , Y


with h5py.File(pjoin(out, 'semg_flexex_smt.h5'), 'w') as f:
    for subj in tqdm(range(1, 41)):

        X , Y= get_data(subj)
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        f.create_dataset('s' + str(subj) + '/X', data=X)
        f.create_dataset('s' + str(subj) + '/Y', data=Y)
       