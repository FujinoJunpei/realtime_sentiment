# coding:utf-8
from pathlib import Path
from typing import *
import torch
from pytorch_pretrained_bert.modeling import BertModel

from functools import partial
from overrides import overrides

import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.iterators import BucketIterator

from allennlp.nn import util as nn_util

from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.training.trainer import Trainer
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy

from allennlp.nn import RegularizerApplicator
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor

@Model.register("rnn_clf")
class RnnClassifier(Model):    
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        dropout: float = 0.,
        label_namespace: str = 'labels',
        initializer: InitializerApplicator = InitializerApplicator(),
        pretrained_path: str = None) -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._ln = nn.LayerNorm(self._classifier_input_dim)
        self._classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()

        initializer(self)

        if pretrained_path:
            with open(pretrained_path, 'rb') as f:
                self.load_state_dict(torch.load(f))
        
    def forward(self, tokens, label=None):
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        encoded_text = self._dropout(self._seq2vec_encoder(embedded_text, mask=mask))
        encoded_text = self._ln(encoded_text)
        logits = self._classification_layer(encoded_text)
        probs = F.softmax(logits, dim=1)
        output_dict = {'logits': logits, 'probs': probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss'] = loss
            self._accuracy(logits, label)
        return output_dict
    def get_metrics(self, reset=False):
        return {'accuracy': self._accuracy.get_metric(reset)}
