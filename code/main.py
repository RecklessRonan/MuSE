import argparse
import random
from unittest import result
import numpy as np
import torch
import torch.nn as nn
import logging
import os
import json
import glob
import time

from transformers import WEIGHTS_NAME, AutoConfig, get_linear_schedule_with_warmup, AdamW, AutoTokenizer, AutoModel
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from sklearn.metrics import accuracy_score, f1_score


from utils import (MTMNERProcessor, MNERProcessor, MNERDataset, ProgressBar, 
                    get_vocabulary, get_entities, SeqEntityScore, json_to_text, MSAProcessor,
                    get_msa_vocabulary, MSADataset)
from models import (CrossReplaceTransformerLayer, CrossReplaceTransformer, 
                    TextEncoder, TextDecoder, ImageEncoder, ImageDecoder,
                    discretized_mix_logistic_loss, LOSS_IT, CRF)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



class SubspaceExchangeMNER(nn.Module):
    def __init__(self, image_encoder, text_decoder, text_encoder, image_decoder, cross_transformer, num_labels, cls_init, 
                        hidden_size, crf_dropout, text_dropout, image_dropout):
        super(SubspaceExchangeMNER, self).__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder
        self.text_encoder = text_encoder
        self.image_decoder = image_decoder
        self.cross_transformer = cross_transformer
        # self.linear = nn.Linear(6272, hidden_size) # (encoded_image_size * encoded_image_size * encoder_dim / max_seq_length, 768)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size*2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.crf_dropout = nn.Dropout(crf_dropout)
        self.text_dropout = nn.Dropout(text_dropout)
        self.image_dropout = nn.Dropout(image_dropout)
        
        self.cls_i = Parameter(torch.rand(1, hidden_size), requires_grad=True)
        self.cls_t = Parameter(torch.rand(1, hidden_size), requires_grad=True)
        # self.share_mlp = nn.Linear(hidden_size, hidden_size)
        if cls_init == 0:
            nn.init.kaiming_normal_(self.cls_i, mode='fan_out')
            nn.init.kaiming_normal_(self.cls_t, mode='fan_out')
        elif cls_init == 1:
            nn.init.kaiming_uniform_(self.cls_i, mode='fan_out')
            nn.init.kaiming_uniform_(self.cls_t, mode='fan_out')
        elif cls_init == 2:
            nn.init.xavier_uniform_(self.cls_i)
            nn.init.xavier_uniform_(self.cls_t)
        elif cls_init == 3:
            nn.init.xavier_normal_(self.cls_i)
            nn.init.xavier_normal_(self.cls_t)
        

    def forward(self, text_input_ids, image_decode, image_raw, caption, length, labels=None, attention_mask=None, token_type_ids=None):
        last_hidden_state, pooler_output = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = self.linear(pooler_output)
        image_generate = self.image_decoder(x=image_decode, h=pooler_output)
        image_encode_embed = self.image_encoder(image_raw) 
        image_encode_embed_view = image_encode_embed.contiguous().view(image_encode_embed.shape[0], -1, image_encode_embed.shape[-1])
        image_encode_embed = self.linear(image_encode_embed)
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.text_decoder(image_encode_embed, caption, length)
        last_hidden_state = self.text_dropout(last_hidden_state)
        image_encode_embed_view = self.image_dropout(image_encode_embed_view)
        cls_t = self.cls_t.repeat(last_hidden_state.shape[0], 1, 1)
        cls_i = self.cls_i.repeat(image_encode_embed_view.shape[0], 1, 1)
        last_hidden_state_view = torch.cat((cls_t, last_hidden_state), dim=1)
        image_encode_embed_view = torch.cat((cls_i, image_encode_embed_view), dim=1)
        sequence_output, image_output = self.cross_transformer(last_hidden_state_view, image_encode_embed_view)
        sequence_output = self.linear2(torch.cat((sequence_output[:, 1:, :], image_output[:, 1:, :]), dim=-1))
        sequence_output = self.crf_dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, outputs, logits


class SubspaceExchangeMSA(nn.Module):
    def __init__(self, image_encoder, text_decoder, text_encoder, image_decoder, cross_transformer, num_labels, cls_init, 
                        hidden_size, crf_dropout, text_dropout, image_dropout):
        super(SubspaceExchangeMSA, self).__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder
        self.text_encoder = text_encoder
        self.image_decoder = image_decoder
        self.cross_transformer = cross_transformer
        # self.linear = nn.Linear(6272, hidden_size) # (encoded_image_size * encoded_image_size * encoder_dim / max_seq_length, 768)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size*2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf_dropout = nn.Dropout(crf_dropout)
        self.text_dropout = nn.Dropout(text_dropout)
        self.image_dropout = nn.Dropout(image_dropout)
        self.cls_i = Parameter(torch.rand(1, hidden_size), requires_grad=True)
        self.cls_t = Parameter(torch.rand(1, hidden_size), requires_grad=True)
        # self.share_mlp = nn.Linear(hidden_size, hidden_size)
        if cls_init == 0:
            nn.init.kaiming_normal_(self.cls_i, mode='fan_out')
            nn.init.kaiming_normal_(self.cls_t, mode='fan_out')
        elif cls_init == 1:
            nn.init.kaiming_uniform_(self.cls_i, mode='fan_out')
            nn.init.kaiming_uniform_(self.cls_t, mode='fan_out')
        elif cls_init == 2:
            nn.init.xavier_uniform_(self.cls_i)
            nn.init.xavier_uniform_(self.cls_t)
        elif cls_init == 3:
            nn.init.xavier_normal_(self.cls_i)
            nn.init.xavier_normal_(self.cls_t)
        

    def forward(self, text_input_ids, image_decode, image_raw, caption, length, labels=None, attention_mask=None, token_type_ids=None):
        # print('text_input_ids', text_input_ids.shape)
        # print(text_input_ids)
        # print('image_decode', image_decode.shape)
        # print(image_decode)
        # print('image_raw', image_raw.shape)
        # print(image_raw)
        # print('caption', caption.shape)
        # print(caption)
        # print('length', length.shape)
        # print(length)
        last_hidden_state, pooler_output = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = self.linear(pooler_output)
        image_generate = self.image_decoder(x=image_decode, h=pooler_output)
        image_encode_embed = self.image_encoder(image_raw) 
        image_encode_embed_view = image_encode_embed.contiguous().view(image_encode_embed.shape[0], -1, image_encode_embed.shape[-1])
        image_encode_embed = self.linear(image_encode_embed)
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.text_decoder(image_encode_embed, caption, length)
        cls_t = self.cls_t.repeat(last_hidden_state.shape[0], 1, 1)
        cls_i = self.cls_i.repeat(image_encode_embed_view.shape[0], 1, 1)
        last_hidden_state = self.text_dropout(last_hidden_state)
        image_encode_embed_view = self.image_dropout(image_encode_embed_view)
        last_hidden_state = torch.cat((cls_t, last_hidden_state), dim=1)
        image_encode_embed_view = torch.cat((cls_i, image_encode_embed_view), dim=1)
        sequence_output, image_output = self.cross_transformer(last_hidden_state, image_encode_embed_view)
        sequence_output = self.linear2(torch.cat((sequence_output[:, 0, :], image_output[:, 0, :]), dim=-1))
        sequence_output = torch.squeeze(sequence_output, 1)
        sequence_output = self.crf_dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, logits


def bulid_model(resnet_pretrained_dir, text_decoder_dim, text_decoder_embed_dim, text_vocab_size,
                 text_decoder_attention_dim, device, text_encoder, d_model, nhead, theta, num_labels, args):
    image_encoder = ImageEncoder(resnet_pretrained_dir, encoded_image_size=8)
    text_decoder = TextDecoder(text_decoder_attention_dim, text_decoder_embed_dim, text_decoder_dim, text_vocab_size,
                                encoder_dim=args.hidden_size, dropout=0.5, device=device)
    text_encoder = TextEncoder(text_encoder, use_xlmr=args.use_xlmr)
    image_decoder = ImageDecoder(nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3)
    cross_transformer_layer = CrossReplaceTransformerLayer(d_model, nhead, theta, skip_connection=args.skip_connection,
                                                            use_quantile=args.use_quantile, dropout=args.cross_dropout)
    cross_transformer = CrossReplaceTransformer(encoder_layer=cross_transformer_layer, num_layers=args.num_layers, 
                                                replace_start=args.replace_start, replace_end=args.replace_end, norm=None)
    if args.task not in ['mvsa-single', 'mvsa-multiple']:
        model = SubspaceExchangeMNER(image_encoder=image_encoder, text_decoder=text_decoder,
                                        text_encoder=text_encoder, image_decoder=image_decoder,
                                        cross_transformer=cross_transformer, num_labels=num_labels, cls_init=args.cls_init,
                                        hidden_size=args.hidden_size, crf_dropout=args.crf_dropout,
                                        text_dropout=args.text_dropout, image_dropout=args.image_dropout)
    else:
        model = SubspaceExchangeMSA(image_encoder=image_encoder, text_decoder=text_decoder,
                                        text_encoder=text_encoder, image_decoder=image_decoder,
                                        cross_transformer=cross_transformer, num_labels=num_labels, cls_init=args.cls_init,
                                        hidden_size=args.hidden_size, crf_dropout=args.crf_dropout,
                                        text_dropout=args.text_dropout, image_dropout=args.image_dropout)
    return model


def evaluate(args, model, eval_dataset, LOSS_TI, cross_entropy, prefix="", flag=None):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # if flag == 'test':
    #     args.eval_batch_size = args.test_batch_size
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=args.drop_last)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"text_input_ids": batch[0], "image_decode": batch[5], "image_raw": batch[3], 
                        "caption": batch[6], "length": batch[7], "labels": batch[4], "attention_mask": batch[1], "token_type_ids": batch[2]}
            image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, outputs, logits = model(**inputs)
            loss_ti = LOSS_TI(batch[5], image_generate)
            loss_it = LOSS_IT(predictions, alphas, encoded_captions, decode_lengths, cross_entropy)
            loss_task = outputs[0]
            # print('loss_ti:', loss_ti, ', loss_it:', loss_it, ', loss_task:', loss_task)
            tmp_eval_loss = calculate_multi_loss(loss_it=loss_it, loss_ti=loss_ti, loss_task=loss_task, args=args)
            pbar(step, {'loss': tmp_eval_loss.item()})
            tags = model.crf.decode(outputs[1], inputs['attention_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[8].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results, eval_loss


def evaluate_msa(args, model, eval_dataset, LOSS_TI, cross_entropy, prefix="", flag=None):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # if flag == 'test':
    #     args.eval_batch_size = args.test_batch_size
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=args.drop_last)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module

    all_labels = []
    all_preds = []
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"text_input_ids": batch[0], "image_decode": batch[5], "image_raw": batch[3], 
                        "caption": batch[6], "length": batch[7], "labels": batch[4], "attention_mask": batch[1], "token_type_ids": batch[2]}
            image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, logits = model(**inputs)
            loss_ti = LOSS_TI(batch[5], image_generate)
            loss_it = LOSS_IT(predictions, alphas, encoded_captions, decode_lengths, cross_entropy)
            loss_task = cross_entropy(logits, batch[4]) * args.max_seq_length
            # print('loss_ti:', loss_ti, ', loss_it:', loss_it, ', loss_task:', loss_task)
            tmp_eval_loss = calculate_multi_loss(loss_it=loss_it, loss_ti=loss_ti, loss_task=loss_task, args=args)
            pbar(step, {'loss': tmp_eval_loss.item()})
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(batch[4].tolist())
            all_preds.extend(preds.tolist())
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)
    eval_loss = eval_loss / nb_eval_steps
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    results = {'f1': f1, 'accuracy': accuracy}
    logger.info("\n")
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results, eval_loss


def predict(args, model, test_dataset, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, drop_last=args.drop_last)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"text_input_ids": batch[0], "image_decode": batch[5], "image_raw": batch[3], 
                        "caption": batch[6], "length": batch[7], "labels": batch[4], "attention_mask": batch[1], "token_type_ids": batch[2]}
            image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, outputs, logits = model(**inputs)
            # print('logits', logits)
            tags = model.crf.decode(outputs[1], inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        # print('preds', preds)
        # print('args.id2label', args.id2label)
        # print('args.markup', args.markup)
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')


def calculate_multi_loss(loss_it, loss_ti, loss_task, args):
    loss_final = args.alpha * loss_it / (args.max_seq_length) \
                         + args.beta * loss_ti / (args.ti_crop_size * args.ti_crop_size) \
                         + args.sigma * loss_task / (args.max_seq_length) 
    return loss_final

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data (except image) dir")
    parser.add_argument("--image_dir",
                        default=None,
                        type=str,
                        help="The input image dir")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--alpha",
                        default=0.001,
                        type=float,
                        help="The weight for image caption loss.")
    parser.add_argument("--beta",
                        default=0.001,
                        type=float,
                        help="The weight for text-to-image generation loss.")
    parser.add_argument("--sigma",
                        default=1.0,
                        type=float,
                        help="The weight for task loss.")
    parser.add_argument("--theta",
                        default=0.2,
                        type=float,
                        help="The threshold for cross-replacing attention.")
    parser.add_argument("--bert_type",
                        default='uncased',
                        type=str,
                        help="The type for Bert, such as uncased, cased, chinese, xlmr-base.")
    parser.add_argument("--resnet_pretrained_dir",
                        default='/home/hadoop-aipnlp/cephfs/data/zhurenyu/kg-multimodal/MNER/baselines/RpBERT-master/pretrained/resnet/resnet152-b121ed2d.pth',
                        type=str,
                        help="The checkpoint dir for pretrained resnet.")
    parser.add_argument('--crop_size', type=int,
                        default=224, help='crop size of image')
    parser.add_argument('--ti_crop_size', type=int,
                        default=32, help='crop size of image in text-to-image generation')
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--per_gpu_train_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--num_workers', type=int,
                        default=8, help='the parallel workers for dataloader')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--predict_checkpoints",type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--drop_last", action="store_true",
                        help="Whether to drop the last batch.")
    parser.add_argument('--num_layers', type=int,
                        default=6, help='the layers of cross-transformer')
    parser.add_argument('--replace_start', type=int,
                        default=0, help='the start layer of replace in cross-transformer')
    parser.add_argument('--replace_end', type=int,
                        default=5, help='the end layer of replace in cross-transformer')
    parser.add_argument('--cls_init', type=int,
                        default=0, help='the initilization of [cls] in cross-transformer')
    parser.add_argument("--hidden_size", default=768, type=int,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--crf_dropout", default=0.5, type=float,
                        help="dropout for crf classifier.")
    parser.add_argument("--text_dropout", default=0.0, type=float,
                        help="dropout for text embedding.")
    parser.add_argument("--image_dropout", default=0.0, type=float,
                        help="dropout for image embedding.")
    parser.add_argument("--cross_dropout", default=0.1, type=float,
                        help="dropout for cross-transformer.")
    parser.add_argument("--skip_connection", action="store_true",
                        help="Whether to add skip connection on cross-transformer")
    parser.add_argument("--use_quantile", action="store_true",
                        help="Whether to use percentage replacing on cross-transformer")
    parser.add_argument("--load_text_checkpoint", action="store_true",
                        help="Whether to load pretrained text checkpoint")
    parser.add_argument("--load_image_checkpoint", action="store_true",
                        help="Whether to load pretrained image checkpoint")

    
    args = parser.parse_args()
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.task == 'twitter15':
        if args.data_dir is None:
            args.data_dir = 'your url'
        if args.image_dir is None:
            args.image_dir = 'your url'
        text_checkpoint = 'your url'
        image_checkpoint = 'your url'
    elif args.task == 'twitter17':
        if args.data_dir is None:
            args.data_dir = 'your url'
        if args.image_dir is None:
            args.image_dir = 'your url'
        text_checkpoint = 'your url'
        image_checkpoint = 'your url'
    elif args.task == 'mt-product':
        if args.data_dir is None:
            args.data_dir = 'your url'
        if args.image_dir is None:
            args.image_dir = 'your url'
    elif args.task == 'mvsa-single':
        if args.data_dir is None:
            args.data_dir = 'your url'
        if args.image_dir is None:
            args.image_dir = 'your url'
        text_checkpoint = 'your url'
        image_checkpoint = 'your url'
    elif args.task == 'mvsa-multiple':
        if args.data_dir is None:
            args.data_dir = 'your url'
        if args.image_dir is None:
            args.image_dir = 'your url'
        text_checkpoint = 'your url'
        image_checkpoint = 'your url'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task in ['twitter15', 'twitter17']:  # MNER
        processor = MNERProcessor()
        args.drop_last = True
    elif args.task in ['mt-product']:
        processor = MTMNERProcessor()
        args.bert_type = 'chinese'
        args.drop_last = True
    elif args.task in ['mvsa-single', 'mvsa-multiple']:
        processor = MSAProcessor()
        args.drop_last = True
    
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}

    logger.info("label_list: {}, length: {}".format(label_list, len(label_list)))


    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    if args.bert_type == 'uncased':
        bert_dir = 'your url'
        args.use_xlmr = False
    elif args.bert_type == 'cased':
        bert_dir = 'your url'
        args.use_xlmr = False
    elif args.bert_type == 'chinese':
        bert_dir = 'your url'
        args.use_xlmr = False
    elif args.bert_type == 'xlmr-base':
        bert_dir = 'your url'
        args.use_xlmr = True
    
    if args.task not in ['mvsa-single', 'mvsa-multiple']:
        vocabulary = get_vocabulary(train_examples) # only use train examples to construct vocabulary of image caption
    else:
        vocabulary = get_msa_vocabulary(train_examples, args.image_dir)
    text_vocab_size = len(vocabulary)
    logger.info(" The size of vocabulary = %d", text_vocab_size)

    bert_config = AutoConfig.from_pretrained(bert_dir)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModel.from_pretrained(bert_dir, config=bert_config)
    model = bulid_model(resnet_pretrained_dir=args.resnet_pretrained_dir, text_decoder_dim=512, text_decoder_embed_dim=512, text_vocab_size=text_vocab_size,
                 text_decoder_attention_dim=512, device=args.device, text_encoder=bert_model,
                 d_model=768, nhead=12, theta=args.theta, num_labels=num_labels, args=args)

    model_dict = model.state_dict()
    if args.load_text_checkpoint:
        text_all_dict = torch.load(text_checkpoint)
        # print('text dict')
        # print(text_all_dict)
        text_decoder_dict = {k: v for k, v in text_all_dict.items() if 'text_decoder' in k}
        model_dict.update(text_decoder_dict)
    if args.load_image_checkpoint:
        image_all_dict = torch.load(image_checkpoint)
        # print('image dict')
        # print(image_all_dict)
        image_decoder_dict = {k: v for k, v in image_all_dict.items() if 'image_decoder' in k}
        model_dict.update(image_decoder_dict)
    model.load_state_dict(model_dict)
    model.to(args.device)
    if args.task not in ['mvsa-single', 'mvsa-multiple']:
        train_dataset = MNERDataset(examples=train_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
        dev_dataset = MNERDataset(examples=dev_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
        test_dataset = MNERDataset(examples=test_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
    else:
        train_dataset = MSADataset(examples=train_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
        dev_dataset = MSADataset(examples=dev_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
        test_dataset = MSADataset(examples=test_examples, label_list=label_list, max_seq_length=args.max_seq_length, 
                                        tokenizer=bert_tokenizer, crop_size=args.crop_size, path_img=args.image_dir, ti_crop_size=args.ti_crop_size,
                                        vocabulary=vocabulary, use_xlmr=args.use_xlmr)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, 
                                    num_workers=args.num_workers, drop_last=args.drop_last)

    LOSS_TI = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    cross_entropy = nn.CrossEntropyLoss()


    logger.info("Args: {}".format(args))
    

    if args.do_train:
        summary_dir = '{}summary_{}'.format(args.output_dir, int(time.time()))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        writer = SummaryWriter(summary_dir)
        logger.info("Summary dir: {}".format(summary_dir))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.warmup_steps = int(t_total * args.warmup_proportion)

        if args.task in ['mvsa-single', 'mvsa-multiple']:
            optimizer_grouped_parameters = model.parameters()
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            all_optimizer = list(model.named_parameters())
            crf_param_optimizer = list(model.crf.named_parameters())
            linear_param_optimizer = list(model.classifier.named_parameters())
            other_optimizer = []
            for op in all_optimizer:
                # print(op[0])
                if 'crf' not in op[0] and 'classifier' not in op[0]:
                    other_optimizer.append(op)
            # print('model', model.parameters())
            # print('crf', crf_param_optimizer)
            # print('linear', linear_param_optimizer)
            # print('other', other_optimizer)
            optimizer_grouped_parameters = [
                {'params': [p for n, p in other_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in other_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        tr_loss, logging_loss = 0.0, 0.0
        global_step = 0
        loss_step = 0
        model.zero_grad()

        best_eval_results = 0
        best_eval_loss = float('inf')

        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
        for epoch in range(int(args.num_train_epochs)):
            pbar.reset()
            pbar.epoch_start(current_epoch=epoch)
            for step, batch in enumerate(train_dataloader):
                # Skip past any already trained steps if resuming training
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"text_input_ids": batch[0], "image_decode": batch[5], "image_raw": batch[3], 
                            "caption": batch[6], "length": batch[7], "labels": batch[4], "attention_mask": batch[1], "token_type_ids": batch[2]}
                if args.task not in ['mvsa-single', 'mvsa-multiple']:
                    image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, outputs, logits = model(**inputs)
                    loss_task = outputs[0]
                else:
                    image_generate, predictions, encoded_captions, decode_lengths, alphas, sort_ind, logits = model(**inputs)
                    # print('logits', logits.shape)
                    # print('batch[4]', batch[4].shape)
                    loss_task = cross_entropy(logits, batch[4]) * args.max_seq_length
                    # preds = torch.argmax(logits, dim=-1)
                    # print('preds', preds.shape)
                    # print(preds.tolist())
                    # print('labels', batch[4].shape)
                    # print(batch[4].tolist())

                loss_ti = LOSS_TI(batch[5], image_generate)
                # print('decode_lengths', decode_lengths)
                loss_it = LOSS_IT(predictions, alphas, encoded_captions, decode_lengths, cross_entropy)
                # print('loss_ti:', loss_ti, ', loss_it:', loss_it, ', loss_task:', loss_task)
                loss_final = calculate_multi_loss(loss_it=loss_it, loss_ti=loss_ti, loss_task=loss_task, args=args)
                writer.add_scalar('loss_ti', loss_ti, loss_step)
                writer.add_scalar('loss_it', loss_it, loss_step)
                writer.add_scalar('loss_task', loss_task, loss_step)
                writer.add_scalar('loss_final', loss_final, loss_step)
                loss_step += 1

                loss_final.backward()
                pbar(step, {'loss': loss_final.item()})
                tr_loss += loss_final.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.evaluate_during_training:
                        # Log metrics
                        print(" ")
                        if args.local_rank == -1:
                            # Only evaluate when single GPU otherwise metrics may not average well
                            if args.task not in ['mvsa-single', 'mvsa-multiple']:
                                results, eval_loss = evaluate(args, model, dev_dataset, LOSS_TI, cross_entropy)
                                eval_results = results['f1']
                            else:
                                results, eval_loss = evaluate_msa(args, model, dev_dataset, LOSS_TI, cross_entropy)
                                eval_results = results['accuracy']
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        bert_tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        if eval_results > best_eval_results:
                            best_eval_results = eval_results
                            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                                os.makedirs(args.output_dir)
                            logger.info("Saving best eval result model checkpoint to %s", args.output_dir)
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.bin"))
                            torch.save(args, os.path.join(args.output_dir, "training_args_best.bin"))
                            bert_tokenizer.save_vocabulary(args.output_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer_best.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler_best.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", args.output_dir)
                        
                        if eval_loss <  best_eval_loss:
                            best_eval_loss = eval_loss
                            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                                os.makedirs(args.output_dir)
                            logger.info("Saving best eval loss model checkpoint to %s", args.output_dir)
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best_loss.bin"))
                            torch.save(args, os.path.join(args.output_dir, "training_args_best_loss.bin"))
                            bert_tokenizer.save_vocabulary(args.output_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer_best_loss.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler_best_loss.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", args.output_dir)

            logger.info("\n")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_last.bin"))
        torch.save(args, os.path.join(args.output_dir, "training_args_last.bin"))
        bert_tokenizer.save_vocabulary(args.output_dir)
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer_last.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler_last.pt"))
        logger.info("Saving optimizer and scheduler states to %s", args.output_dir)
        
    # Evaluation
    # results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint = args.output_dir
        # if args.eval_all_checkpoints:
        #     checkpoints = list(
        #         os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        #     )
        #     logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # for checkpoint in checkpoints:
        # global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        # prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        model = bulid_model(resnet_pretrained_dir=args.resnet_pretrained_dir, text_decoder_dim=512, 
                text_decoder_embed_dim=512, text_vocab_size=text_vocab_size,
                text_decoder_attention_dim=512, device=args.device, text_encoder=bert_model,
                d_model=768, nhead=12, theta=args.theta, num_labels=num_labels, args=args)
        logger.info("Test on best eval model")      
        model.load_state_dict(torch.load(os.path.join(checkpoint, "model_best.bin")))
        prefix = 'test best'
        model.to(args.device)
        if args.task not in ['mvsa-single', 'mvsa-multiple']:
            result, _ = evaluate(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        else:
            result, _ = evaluate_msa(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        # if global_step:
        #     result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        # results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results_best.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        
        logger.info("Test on best eval loss model")      
        model.load_state_dict(torch.load(os.path.join(checkpoint, "model_best_loss.bin")))
        prefix = 'test best_loss'
        model.to(args.device)
        if args.task not in ['mvsa-single', 'mvsa-multiple']:
            result, _ = evaluate(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        else:
            result, _ = evaluate_msa(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        # if global_step:
        #     result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        # results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results_best_loss.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        logger.info("Test on last eval model")      
        model.load_state_dict(torch.load(os.path.join(checkpoint, "model_last.bin")))
        # model.to(args.device)
        prefix = 'test last'
        if args.task not in ['mvsa-single', 'mvsa-multiple']:
            result, _ = evaluate(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        else:
            result, _ = evaluate_msa(args, model, test_dataset, LOSS_TI, cross_entropy, prefix=prefix, flag='test')
        # if global_step:
        #     result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        # results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results_last.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))


    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = bulid_model(resnet_pretrained_dir=args.resnet_pretrained_dir, text_decoder_dim=512, 
                 text_decoder_embed_dim=512, text_vocab_size=text_vocab_size,
                 text_decoder_attention_dim=512, device=args.device, text_encoder=bert_model,
                 d_model=768, nhead=12, theta=args.theta, num_labels=num_labels, args=args)
            model.load_state_dict(torch.load(os.path.join(checkpoint, "model.bin")))
            model.to(args.device)
            predict(args, model, test_dataset, prefix=prefix)   
        

if __name__ == "__main__":
    main()
