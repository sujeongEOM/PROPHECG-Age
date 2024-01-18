# PROPHECG-Age
PRediction Of PHenotypes using ElectroCardioGraphy-Age  
Goal : ``Predict ECG-age using raw ECG waveform and deep neural network``
     

# ECG Data processing
- Raw ECG waveform with 500Hz / 10 sec / 12 leads
- Use only 8 leads (I, II, V1-V6) as other 4 leads are computed using these 8 leads
- Input shape : (5000, 8)


# Model Train & Test
## Model architecture  
Below image shows the architecture of 1 dimensional residual block neural network based convolutional neural network used for the age prediction. As the purpose is predicting age, this is a regression task. Final output after dense layer would be **AGE**. 
<p align="center">
<img src = "https://github.com/sujeongEOM/PROPHECG-Age/assets/81948366/5c67d5c7-4b8b-4d55-87dc-9c1ac5dc0f8d" width="70%" height="70%">
</p>
Ref) Lima et al., Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 



## Folder contents


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

