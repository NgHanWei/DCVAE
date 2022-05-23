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
    description='Preprocessor for sEMG Data')
parser.add_argument('source', type=str, default='D:/sEMG_data/sEMG-dataset/filtered/mat/', help='Path to raw sEMG data')
parser.add_argument('target', type=str, default='D:/eeg_vrae/pre-processed-semg', help='Path to pre-processed sEMG data')
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
    # # Obtain Ulnar
    # X11 = X[:,34*250:40*250]
    # X12 = X[:,168*250:174*250]
    # X13 = X[:,(134*2+34)*250:(134*2+40)*250]
    # X14 = X[:,(134*3+34)*250:(134*3+40)*250]
    # X15 = X[:,(134*4+34)*250:(134*4+40)*250]    
    # # Obtain Radial
    # X16 = X[:,44*250:50*250]
    # X17 = X[:,178*250:184*250]
    # X18 = X[:,(134*2+44)*250:(134*2+50)*250]
    # X19 = X[:,(134*3+44)*250:(134*3+50)*250]
    # X20 = X[:,(134*4+44)*250:(134*4+50)*250]
    # # Obtain Grip
    # X21 = X[:,54*250:60*250]
    # X22 = X[:,188*250:194*250]
    # X23 = X[:,(134*2+54)*250:(134*2+60)*250]
    # X24 = X[:,(134*3+54)*250:(134*3+60)*250]
    # X25 = X[:,(134*4+54)*250:(134*4+60)*250]
    # # Obtain Abduction
    # X26 = X[:,64*250:70*250]
    # X27 = X[:,198*250:204*250]
    # X28 = X[:,(134*2+64)*250:(134*2+70)*250]
    # X29 = X[:,(134*3+64)*250:(134*3+70)*250]
    # X30 = X[:,(134*4+64)*250:(134*4+70)*250]
    # # Obtain Adduction
    # X31 = X[:,74*250:80*250]
    # X32 = X[:,208*250:214*250]
    # X33 = X[:,(134*2+74)*250:(134*2+80)*250]
    # X34 = X[:,(134*3+74)*250:(134*3+80)*250]
    # X35 = X[:,(134*4+74)*250:(134*4+80)*250]
    # # Obtain Supination
    # X36 = X[:,84*250:90*250]
    # X37 = X[:,218*250:224*250]
    # X38 = X[:,(134*2+84)*250:(134*2+90)*250]
    # X39 = X[:,(134*3+84)*250:(134*3+90)*250]
    # X40 = X[:,(134*4+84)*250:(134*4+90)*250]
    # # Obtain Pronation
    # X41 = X[:,94*250:100*250]
    # X42 = X[:,228*250:234*250]
    # X43 = X[:,(134*2+94)*250:(134*2+100)*250]
    # X44 = X[:,(134*3+94)*250:(134*3+100)*250]
    # X45 = X[:,(134*4+94)*250:(134*4+100)*250]
    # X_new = np.array([X1, X2, X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39,X40,X41,X42,X43,X44,X45])
    X_new = np.array([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10])
    print(X_new.shape)

    # X_new = X_new * 3000

    # Obtain Labels 0 - Extension, 1 - Flexion
    Y = np.array([0,0,0,0,0,1,1,1,1,1])
    print(Y.shape)

    return X_new , Y


with h5py.File(pjoin(out, 'semg_flexex_smt.h5'), 'w') as f:
    for subj in tqdm(range(1, 41)):

        X , Y= get_data(subj)
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        print(X.shape)
        print(Y.shape)
        f.create_dataset('s' + str(subj) + '/X', data=X)
        f.create_dataset('s' + str(subj) + '/Y', data=Y)
       