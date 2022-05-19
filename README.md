# MuSE

This is the implementation of paper ***Exchanging-based Multimodal Fusion with Transformer***

We provide our data in [Google Drive](https://drive.google.com/drive/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn?usp=sharing).

## Setup

This implemetation is based on PyTorch. To run the code, you need the following dependencies:

- PyTorch==1.5.1


## Repository structure

```python
|-- code
    |-- configs # configurations for code summarization (cs) and code clone detection (ccd)
    |   |-- config_ccd.yml
    |   |-- config_cs.yml
    |-- features # store the processed features for 4 datasets
    |   |-- BCB
    |   |-- BCB-F
    |   |-- CSN
    |   |-- TLC
```

## Run pipeline

We use the code summarization task as example. The code clone detection task follows the similar pipeline. We conduct all experiments on two Tesla V100 GPUs.



## Attribution

Parts of this code are based on the following repositories:

- [CodeBERT](https://github.com/microsoft/CodeBERT)
