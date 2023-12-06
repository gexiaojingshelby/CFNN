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
Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and NBA dataset  
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
