# Coarse-Fine Nested Network for Weakly-Supervised Group Activity Recognition
## Dependencies
Python>=3.6  
PyTorch=1.8.1 and corresponding torchvision=0.9.0 (if use other versions, note the differences in the torch.nn.MultiheadAttention parameter list)  
Pillow>=8.1.1  
PyYAML>=5.4  
easydict  
numpy  
tensorboardX>=2.0  
## Prepare Datasets
### Volleyball dataset
Download Volleyball dataset from: [Volleyball dataset](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing).
### NBA dataset  
Due to the copyright restriction, this dataset is available upon [request](https://ruiyan1995.github.io/SAM.html).
## Train
### Volleyball dataset
bash train_volleyball.sh
### NBA dataset
bash train_nba.sh
## Validate
### Volleyball dataset
bash validate_volleyball.sh
### NBA dataset
bash validate_nba.sh
## Reference
Our code is based on the open-source project of [DFWSGAR](https://github.com/dk-kim/DFWSGAR).
