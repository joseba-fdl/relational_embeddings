# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

######### GPUtan MARTXAN JARTZEKO ########
# marin: source /var/python3envs/transformers-2.7.0/bin/activate.csh
# setenv CUDA_VISIBLE_DEVICES 0 # setenv CUDA_VISIBLE_DEVICES 1

import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, f1_score, recall_score

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torch.nn as nn
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
#from utils_classification import convert_examples_to_features
from utils_classification_ensemble import convert_examples_to_features
from transformers.data.processors.utils import SingleSentenceClassificationProcessor as Processor



try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class EnsembleaForSequenceClassification(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, config, rel_emb_size=20, num_labels=3, dropautProb=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir if cache_dir else None,
        )
            
        self.dropout = nn.Dropout(dropautProb)
        self.classifier = nn.Linear(config.hidden_size+rel_emb_size, num_labels)
        #self.classifier = nn.Linear(768+rel_emb_size, num_labels)

        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        rel_emb=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,)
            

        pooled_output = outputs[1] ## pooled output ## CLS embedding ## Finetunninga eta geroko balioak!!!
        pooled_output = torch.cat([pooled_output, rel_emb], -1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits, pooled_output)
        '''return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )'''

#kk=EnsembleaForSequenceClassification("bert-base-multilingual-cased",None,config='/ikerlariak/jfernandezde010/emnlp/sailkatzaileak/1-transformers/emaitzak_se_bert-base-multilingual-cased/Abortion')
