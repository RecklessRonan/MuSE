# MuSE

This is the implementation of paper ***Exchanging-based Multimodal Fusion with Transformer***

We provide our collected dataset (MT-Product) in anonymous [Google Drive](). 

And the other public datasets can be downloaded from: 

- [twitter15 and twitter17](https://github.com/jefferyYu/UMT)

- [mvsa-single and mvsa-multiple](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/).


## Setup

This implemetation is based on Python=3.7. To run the code, you need the following dependencies:

- torch==1.11.0
- transformers==4.18.0
- torchvision==0.12.0
- spacy==3.3.0
- scikit-learn==1.0.2
- tensorboard==1.13.1


## Repository structure

```python
|-- code
    |-- main.py
    |-- utils.py
    |-- models.py
|-- data
    |-- MT-Product
|-- requirements.txt
```

## Run pipeline



## Attribution

Parts of this code are based on the following repositories:

- [UMT](https://github.com/jefferyYu/UMT)

- [NIC](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

- [PixelCNN++](https://github.com/pclucas14/pixel-cnn-pp)
