# DualChainVAE
Codes for DualChainVAE Pytorch Implementation for extraction of spatiotemporal features. A pytorch implementation of a vanilla VAE counterpart is included.

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

### DualChainVAE
The DualChainVAE can be found implemented in `dual_vae_torch.py`.
```
usage: python dual_vae_torch.py [-subj SUBJ] [-epochs EPOCHS] [-features FEATURES]

Arguments:
-subj SUBJ              Set the subject number to run feature extraction on
-epochs EPOCHS          Set the number of epochs for which to train the VAE
-features FEATURES      Set the desired number of features to extract from the signal

```
