# DualChainVAE
Codes for DualChainVAE Pytorch Implementation for extraction of spatiotemporal features. A pytorch implementation of a vanilla VAE counterpart is included. The dataset used for spatiotemporal feature extraction is an EEG Dataset from Korea University consisting of 54 subjects.

## Resources
Raw Dataset: http://gigadb.org/dataset/100542

## Instructions
### Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:

```
pip install -r requirements.txt
```

### Obtain the raw dataset
Download the raw dataset from the resources above, and save them to the same `$source` folder. To conserve space, you may only download files that ends with `EEG_MI.mat`.

### Pre-process raw dataset
The following command will read the raw dataset from the `$source` folder, and output the pre-processed data `KU_mi_smt.h5` into the `$target` folder.

```
python preprocess_h5_smt.py $source $target
```

## Training and Evaluation

### Traditional VAE
The traditional VAE can be found implemented in `vae_torch.py`.
```
usage: python vae_torch.py [-subj SUBJ] [-epochs EPOCHS] [-features FEATURES]

Arguments:
-subj SUBJ              Set the subject number to run feature extraction on
-epochs EPOCHS          Set the number of epochs for which to train the VAE
-features FEATURES      Set the desired number of features to extract from the signal

```

Final results may be found in the following files:
```
vae_output_LDA.txt          Reports the Linear Discriminant Analysis Score for the extracted features according to the true class
vae_output_NLL.txt          Reports the test negative log-likeilhood approximated by the total loss
vae_output_recon.txt        Reports the test reconstruction loss by the final trained model

vae_torch.pt                Final saved model weights
```

### DualChainVAE
The DualChainVAE can be found implemented in `dual_vae_torch.py`.
```
usage: python dual_vae_torch.py [-subj SUBJ] [-epochs EPOCHS] [-features FEATURES] [-alpha ALPHA] [-beta BETA] [-loss LOSS]

Arguments:
-subj SUBJ              Set the subject number to run feature extraction on
-epochs EPOCHS          Set the number of epochs for which to train the VAE
-features FEATURES      Set the desired number of features to extract from the signal
-alpha ALPHA            Set the alpha hyperparameter for KL Divergence in the first chain
-beta BETA              Set the beta hyperparameter for KL Divergence in the second chain
-loss LOSS              Uses one of three possible loss functions to train the DCVAE model. 
                        Default - Default DCVAE loss. Full - Uses entire reconstruction loss. Indiv - Sum of individual chain losses.

```

Final results may be found in the following files:
```
dual_output_LDA.txt          Reports the Linear Discriminant Analysis Score for the extracted features according to the true class
dual_output_NLL.txt          Reports the test negative log-likeilhood approximated by the total loss
dual_output_recon.txt        Reports the test reconstruction loss by the final trained model

dual_vae_torch.pt            Final saved model weights
```
