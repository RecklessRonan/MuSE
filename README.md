# MuSE

This is the implementation of paper ***Exchanging-based Multimodal Fusion with Transformer***

We provide our collected dataset (MT-Product) in anonymous [Google Drive](https://drive.google.com/drive/u/1/folders/1emWPyba8kF29EgS67ESO81kxIOCQN6Dv). 

And the other public datasets can be downloaded from: 

- [twitter15 and twitter17](https://github.com/jefferyYu/UMT)

- [mvsa-single and mvsa-multiple](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/).


## Setup

This implemetation is based on Python=3.7. To run the code, you need the following dependencies:

- torch==1.11.0
- transformers==4.18.0
- torchvision==0.12.0
- spacy==2.3.7
- scikit-learn==1.0.2
- tensorboard==2.8.0

Or you can simply run:

```
pip install -r requirements.txt
```

## Repository structure

```python
|-- code
    |-- main.py  # the main function to run the code
    |-- utils.py # the preprocess for all datasets
    |-- models.py # the key models in our experiments, including CrossTransformer
    |-- run_example.sh  # the run example
|-- data
    |-- MT-Product # the example data for MT-product texts. For the full data, please download from the google drive in the above.
|-- requirements.txt
```

## Run example

We give the run example of twitter15 dataset: 

```python
TASK_NAME="twitter15"
alpha=0.0001
beta=0.0001
theta=0.1
sigma=1.0
replace_start=1
replace_end=3
cls_init=0
num_layers=6
crf_dropout=0.5
learning_rate=1e-4
crf_learning_rate=1e-4
bert_type='uncased'
cross_dropout=0.2

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --task ${TASK_NAME} \
    --per_gpu_train_batch_size 40 \
    --per_gpu_eval_batch_size 40 \
    --alpha ${alpha} \
    --beta ${beta} \
    --theta ${theta} \
    --output_dir ../outputs/${TASK_NAME}_output/alpha${alpha}_beta${beta}_theta${theta}_sigma${sigma}_rs${replace_start}_re${replace_end}_cls${cls_init}_l${num_layers}_lr${learning_rate}_clr${crf_learning_rate}_${bert_type}_cd${cross_dropout}_last/\
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --evaluate_during_training \
    --num_workers 8 \
    --learning_rate ${learning_rate} \
    --crf_learning_rate ${crf_learning_rate} \
    --num_layers ${num_layers} \
    --replace_start ${replace_start} \
    --replace_end ${replace_end} \
    --cls_init ${cls_init} \
    --crf_dropout ${crf_dropout} \
    --skip_connection \
    --use_quantile \
    --bert_type ${bert_type} \
    --cross_dropout ${cross_dropout}
```

You can change the 'TASK_NAME' to run other datasets. Also, you can try any parameters in the python scripts.

Notice: you need to change all 'your url' in 'main.py' to the real url of datasets and models.


## CrossTransformer

Here is the core of CrossTransformer. For more details, please refer to the 'models.py'.
```python
def _cr_block(self, x1, x2, attn_weight1, attn_weight2):
        cls_weight1 = attn_weight1[:, 0, :]
        cls_weight2 = attn_weight2[:, 0, :]
        x1_mean = torch.mean(x1, dim=-2)
        x2_mean = torch.mean(x2, dim=-2)
        for i in range(cls_weight1.shape[0]):
            if self.use_quantile:
                theta1 = np.quantile(cls_weight1[i][1:].detach().cpu().numpy(), self.theta)
                theta2 = np.quantile(cls_weight2[i][1:].detach().cpu().numpy(), self.theta)
            else:
                theta1 = self.theta
                theta2 = self.theta
            
            for j in range(1, cls_weight1.shape[1]):  # except the first token, namely [cls]
                if cls_weight1[i][j] < theta1:
                    x1[i][j] = x2_mean[i] + x1[i][j] if self.skip_connection else x2_mean[i]
                if cls_weight2[i][j] < theta2:
                    x2[i][j] = x1_mean[i] + x2[i][j] if self.skip_connection else x1_mean[i]
        return x1, x2
```

## Attribution

Parts of this code are based on the following repositories:

- [UMT](https://github.com/jefferyYu/UMT)

- [NIC](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

- [PixelCNN++](https://github.com/pclucas14/pixel-cnn-pp)
