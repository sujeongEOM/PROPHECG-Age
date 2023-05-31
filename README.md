# PROPHECG-Age
PRediction Of PHenotypes using ElectroCardioGraphy in yonsei-Age
     

# ECG Data processing
- 500Hz / 10 sec / 12 leads
- shape = (5000, 12)
- Use only 8 leads (I, II, V1-V6) as other 4 leads are computed using these 8 leads
- final shape as input = (5120, 8) -> (5000, 8)


# Model Train, Test
- 1D CNN Residual block neural network
```
Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
```
## Folder content


- ``train.py``: Script for training the neural network. To train the neural network run:
```bash
$ python train.py SCRIPT_YAML
```


- ``evaluate.py``: Script for generating the neural network predictions on a given dataset.
```bash
$ python evaluate.py SCRIPT_YAML VALID_DATASET
```

- ``model.pth``: Pre-trained model weights trained with about 1.5 million ECGs. 

- ``resnet.py``: Auxiliary module that defines the architecture of the deep neural network.


- ``CustomDataset.py``: Customed Dataset to be put into DataLoader.

- ``script.yaml``: Script with parameters needed for training and validation.  

