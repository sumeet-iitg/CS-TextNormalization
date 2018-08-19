#!/usr/bin/python
# -*- coding:utf-8 -*-


# Portions of this code are adapted from the TensorFlow example materials, which
# retains its original Apache Licenese 2.0.  All other code was developed by
# David Jurgens and Yulia Tsvetkov and has the same license.
#
# Copyright 2017 David Jurgens and The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""
Equilid: Socially-Equitable Language Identification.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import os.path
import re
import argparse
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from glob import glob
from os.path import basename

# import seq2seq_model

from random import shuffle
import os.path
import codecs
import torchtext

import string

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from Seq2seq import seq2seq
from supervised_trainer import SupervisedTrainer
from loss import Perplexity

from fields import SourceField, TargetField
from optim import Optimizer
from checkpoint import Checkpoint
from predictor import Predictor
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train sampleData')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev sampleData')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--train', dest='train_mode', action='store_true')
parser.add_argument('--no-train', dest='train_mode', action='store_false')
parser.set_defaults(train_mode=True)
parser.add_argument('--learning_rate', action='store', dest='learning_rate', default=0.5, help='Learning rate.')
parser.add_argument('--learning_rate_decay_factor', action='store', dest='dr', default=0.99,
                          help='Learning rate decays by this much.')
parser.add_argument('--max_gradient_norm', action='store', dest='max_gradient_norm', default=5.0, help='Clip gradients to this norm.')
parser.add_argument('--batch_size', action='store',dest='batch_size', default=64, help='Batch size to use during training.')
parser.add_argument('--size', action='store', dest='size', default=512, help='Size of each model layer.')

parser.add_argument('--num_layers',action='store', dest='num_layers', default=3, help='Number of layers in Encoder and Decoder.' )
parser.add_argument('--char_vocab_size',action='store', dest='char_vocab_size', default=40000, help='Character vocabulary size.')
parser.add_argument('--lang_vocab_size',action='store', dest='lang_vocab_size', default=40000, help='Language vocabulary size.')
parser.add_argument('--data_dir',action='store', dest='data_dir', default='/tmp', help='Data directory')

# Have the model be loaded from a path relative to where we currently are
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = cur_dir + "/../models/70lang"

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']

# need to have special ids for unk, sos and eos, etc.
_SPECIALS = {'unk_token':'<unk>','sos_token':'<sos>','eos_token':'<eos>','pad_token':'<pad>'}

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [ (60, 11), (100, 26), (140, 36)]

FLAGS = parser.parse_args()

def load_dataset(data_dir, data_type, srcField, tgtField, max_len, select = None):
    """Loads the dataset from all sources in torchtext tabulardataset fmt"""

    # Get all the files of IDs
    # glob finds all the files with a given pattern
    files = [f for f in glob(data_dir + '/*' + data_type + '.ids')]

    # The match the source and target files
    prefices = set()
    for f in files:
        prefices.add(basename(f).split(".")[0])
    loaded_files = []

    for p in prefices:
        if (select is not None) and (not select in p):
            continue
        src = data_dir + '/' + p + '.source.' + data_type + '.ids'
        tgt = data_dir + '/' + p + '.target.' + data_type + '.ids'
        src_tgt_combined = data_dir + '/' + p + '.source_target.' + data_type + '.ids'
        #joining the source and target into a single file for easier time reading with torchtext
        if not os.path.exists(src_tgt_combined):
            with open(src, 'r') as src_fp, open(tgt, 'r') as tgt_fp, open(src_tgt_combined, 'w') as src_tgt_fp:
                src_line = src_fp.readline().strip()
                tgt_line = tgt_fp.readline().strip()
                while src_line and tgt_line:
                    src_tgt_fp.write(src_line + '\t' + tgt_line + '\n')
                    src_line = src_fp.readline().strip()
                    tgt_line = tgt_fp.readline().strip()

        tabularFile = torchtext.data.TabularDataset(path=src_tgt_combined, format='tsv',
                        fields=[('src', srcField),('tgt', tgtField)],
                        filter_pred=lambda x: len(x.src) < max_len)

        # debug tip: write another function here to see if it is getting read properly
        loaded_files.append(tabularFile)
        logger.debug("FileName:{} Example Len:{}".format(src_tgt_combined, len(tabularFile.examples)))

    logger.debug("Files loaded count:{}".format(len(loaded_files)))
    return loaded_files

def get_vocab(vocab_file_path):
    itos = []

    with open(vocab_file_path) as char_fp:
        for line in char_fp:
            itos.append(line.split("\t")[0].strip())

    # setting the default-dict value in stoi for some unknown character
    # stoi = defaultdict(lambda: len(itos))
    stoi = {}
    for i, sym in enumerate(itos):
        stoi[str(sym)] = i
    return itos, stoi

def create_model(sourceVocabClass, targetVocabClass):
    """Create translation model and initialize or load parameters in session."""
    # Prepare src char vocabulary and target vocabulary dataset
    max_len = 50

    # Initialize model
    hidden_size = FLAGS.size
    bidirectional = False
    srcField = SourceField(use_vocab=True)
    tgtField = TargetField(use_vocab=True)

    srcField.vocab = sourceVocabClass
    tgtField.vocab = targetVocabClass

    srcField.set_specials(_SPECIALS)
    tgtField.set_specials(_SPECIALS)

    encoder = EncoderRNN(FLAGS.char_vocab_size,
                         max_len,
                         hidden_size,
                         n_layers=FLAGS.num_layers,
                         bidirectional=bidirectional,
                         variable_lengths=True)
    decoder = DecoderRNN(FLAGS.lang_vocab_size, max_len, hidden_size,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                         eos_id=tgtField.vocab.stoi[tgtField.eos_token], sos_id=tgtField.vocab.stoi[tgtField.sos_token], n_layers=FLAGS.num_layers)

    seq2seqModel = seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seqModel.cuda()

    for param in seq2seqModel.parameters():
        param.data.uniform_(-0.08, 0.08)

    # Prepare loss
    weight = torch.ones(FLAGS.lang_vocab_size)
    loss = Perplexity(weight, tgtField.vocab.stoi[tgtField.pad_token])

    return seq2seqModel, loss, srcField, tgtField

def train(sourceVocabClass, targetVocabClass):
    """Train the Equilid Model from character to language-tagged-token sampleData."""
    # Ensure we have a directory to write to
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    max_len = 50
    seq2seqModel, loss, srcField, tgtField = create_model(sourceVocabClass, targetVocabClass)

    dev_set = load_dataset(FLAGS.data_dir, 'dev',srcField, tgtField, max_len)
    full_train_set = load_dataset(FLAGS.data_dir,'train', srcField, tgtField, max_len)
    assert len(dev_set) == len(full_train_set)

    logger.debug("char itos:{}".format(srcField.vocab.itos))
    logger.debug("char stoi:{}".format(srcField.vocab.stoi))
    logger.debug("lang itos:{}".format(tgtField.vocab.itos))
    logger.debug("lang stoi:{}".format(tgtField.vocab.stoi))

    if torch.cuda.is_available():
        loss.cuda()

    print("Training model")
    t = SupervisedTrainer(loss=loss, batch_size=int(FLAGS.batch_size),
                          checkpoint_every=50,
                          print_every=20, expt_dir=FLAGS.expt_dir)
    optimizer = Optimizer(torch.optim.Adam(seq2seqModel.parameters(), lr=FLAGS.learning_rate), max_grad_norm=FLAGS.max_gradient_norm)

    for i in range(len(full_train_set)):
        seq2seqModel = t.train(seq2seqModel, full_train_set[i],
                      num_epochs=6, dev_data=dev_set[i],
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=FLAGS.resume)
    print("training completed!")

def repair(tokens, predictions):
    """
    Repairs the language prediction sequence if the number of predictions did not
    match the input number of tokens and double-checks that punctuation in the
    input is aligned with the prediction's.  This function is necessary because
    of stochasticity in the LSTM output length and only has much effect
    for very short inputs or very long inputs.
    """

    # If we made a prediction for each token, return that.  NOTE: we could
    # probably do some double checking her to make sure
    # punctiation/hashtag/mention predictions are where they should be
    if len(tokens) == len(predictions):
        return predictions

    # If we only have words (no punctuation), then trunctate to the right number
    # of tokens
    if len(set(predictions)) == 1:
        return predictions[:len(tokens)]

    # See how many languages we estimated
    langs = set([x for x in predictions if len(x) == 3])

    # If we only ever saw one language (easy case), then we've just screwed up
    # the number of tokens so iterate over the tokens and fill in the blanks
    if len(langs) == 1:
        lang = list(langs)[0]

        # This is the output set of assignments, based on realignment
        repaired = []

        # Figure out where we have punctuation in the input
        for i, token in enumerate(tokens):
            if re.fullmatch(r"\p{P}+", token):
                repaired.append('Punct')
            elif re.fullmatch(r"#([\w_]+)", token):
                repaired.append('#Hashtag')
            elif re.fullmatch(r"@([\w_]+)", token):
                repaired.append('@Mention')
            elif (token.startswith('http') and ':' in token) \
                    or token.startswith('pic.twitter'):
                repaired.append('URL')
            else:
                repaired.append(lang)

        # print('%s\n%s\n' % (predictions, repaired))

        return repaired

    else:
        # NOTE: the most rigorous thing to do would be a sequence alignment with
        # something like Smith-Waterman and then fill in the gaps, but this is
        # still likely overkill for the kinds of repair operations we expect

        # This is the output set of assignments, based on realignment
        repaired = []
        n = len(predictions) - 1

        # Figure out where we have non-text stuff in the input as anchor points
        last_anchor = -1
        anchors = []

        rep_anchor_counts = []
        pred_anchor_counts = []

        for pred in predictions:
            prev = 0
            if len(pred_anchor_counts) > 0:
                prev = pred_anchor_counts[-1]
            if len(pred) != 3:
                pred_anchor_counts.append(1 + prev)
            else:
                pred_anchor_counts.append(prev)

        for i, token in enumerate(tokens):
            if re.fullmatch(r"\p{P}+", token):
                repaired.append('Punct')
            elif re.fullmatch(r"#([\w_]+)", token):
                repaired.append('#Hashtag')
            elif re.fullmatch(r"@([\w_]+)", token):
                repaired.append('@Mention')
            elif (token.startswith('http') and ':' in token) \
                    or token.startswith('pic.twitter'):
                repaired.append('URL')
            else:
                repaired.append(None)

        for rep in repaired:
            prev = 0
            if len(rep_anchor_counts) > 0:
                prev = rep_anchor_counts[-1]
            if rep is not None:
                rep_anchor_counts.append(1 + prev)
            else:
                rep_anchor_counts.append(prev)

        for i in range(len(repaired)):
            if repaired[i] is not None:
                continue

            try:
                p = pred_anchor_counts[min(i, len(pred_anchor_counts) - 1)]
                r = rep_anchor_counts[i]
            except IndexError as xcept:
                print(repr(xcept))
                print(i, len(pred_anchor_counts) - 1, min(i, len(pred_anchor_counts) - 1))
                continue

            nearest_lang = 'UNK'

            if p < r:
                # The prediction has fewer anchors than the repair at this
                # point, which means it added too many things, so skip ahead
                for j in range(i + 1, len(predictions)):
                    if pred_anchor_counts[min(j, len(pred_anchor_counts) - 1)] >= p:
                        if len(predictions[j]) == 3:
                            nearest_lang = predictions[j]
                            break

            elif p > r:
                # The prediction skipped some input tokens, so rewind until we
                # have the same number of anchors
                for j in range(min(n, i - 1), -1, -1):
                    if pred_anchor_counts[min(j, n)] <= p:
                        if len(predictions[min(j, n)]) == 3:
                            nearest_lang = predictions[min(j, n)]
                            break
            else:
                # Just search backwards for a language
                for j in range(min(i, n), -1, -1):
                    if len(predictions[j]) == 3:
                        nearest_lang = predictions[j]
                        break

            # For early tokens that didn't get an assignment from a backwards
            # search, search forward in a limited manner
            if nearest_lang is None:
                for j in range(i + 1 + anchors[i], min(n + 1, i + 5 + anchors[i])):
                    if len(predictions[j]) == 3:
                        nearest_lang = predictions[j]

            repaired[i] = nearest_lang

        # print('%s\n%s\n' % (predictions, repaired))

        return repaired


cjk_ranges = [
    {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},
    {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},
    {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},
    {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},
    {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},
    {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
    {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
]

try:
    cjk_ranges.extend([
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")},
        {"from": ord(u"\u0002f800"), "to": ord(u"\u0002fa1f")},
    ])
except TypeError as e:
    print('Unable to load extended unicode ranges for CJK character set, ' +
          'some CJK language identification results may be unreliable.')

hangul_ranges = [
    {"from": ord(u"\uAC00"), "to": ord(u"\uD7AF")},
]


def is_cjk(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in cjk_ranges])


def is_hangul(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in hangul_ranges])


CJK_PROXY = str(ord(u"\u4E00"))
HANGUL_PROXY = str(ord(u"\uAC00"))


def to_token_ids(text, char_to_id, UNK_ID):
    """
    Converts input text into its IDs based on a defined vocabularly.
    """
    ids = []
    for c in text:
        # The CJK and Hangul_Syllable unicode blocks are each collapsed into
        # single proxy characters since they are primarily used with a single
        # language and, because these blocks are huge, this saves significant
        # space in the model's lookup table.
        if is_cjk(c):
            c = CJK_PROXY
        elif is_hangul(c):
            c = HANGUL_PROXY
        else:
            c = str(ord(c))
        ids.append(char_to_id.get(c, UNK_ID))
    return ids


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file."""

    # NOTE: the sampleData-to-int conversion uses a +4 offset for indexing due to
    # the starting vocabulary.  We prepend the rev_vocab here to recognize
    # this
    rev_vocab = list(_START_VOCAB)

    with open(vocabulary_path, "rb") as f:
        for line in f:
            rev_vocab.append(line.split("\t")[0].strip())

    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab

def get_langs(text):
    token_langs = classify(text)
    langs = set([x for x in token_langs if len(x) == 3])
    return langs

def load_model():
    checkpoint_path = ""
    if not FLAGS.load_checkpoint is None:
        checkpoint_path = os.path.join(FLAGS.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, FLAGS.load_checkpoint)
    else:
        checkpoint_path = Checkpoint.get_latest_checkpoint(FLAGS.expt_dir)
    logging.info("loading checkpoint from {}".format(checkpoint_path))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    # these are vocab classes with members stoi and itos
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    classifier = (seq2seq, input_vocab, output_vocab)

    return classifier

# The lazily-loaded classifier which is a tuple of the model, input_vocab and output_vocab
classifier = None
# The predictor class constructed on the classifier, input and output vocab.
predictor = None

def classify(seq, langItos):
    """

    :param text: text to classify is by default treated as unicode in Python 3
    :param langVocab: dict of 'langId'->'langCode'
    :return: a sequence of language codes
    """

    global classifier
    global predictor

    if classifier is None:
        # Prediction uses a small batch size
        FLAGS.batch_size = 1
        classifier = load_model()

    # Unpack the classifier into the things we need
    seq2seqModel, char_vocab, lang_vocab = classifier

    if predictor is None:
        predictor = Predictor(seq2seqModel, char_vocab, lang_vocab)

    predicted_labels = predictor.predict(seq)

    # Ensure we have predictions for each token
    # predictions = repair(text.split(), predicted_labels)
    predictions = []
    print("predictions:".format(predicted_labels))
    for pred in predictions:
        predictions.append(langItos[int(pred)])

    return predictions

# def predict():
#     # NB: is there a safer way to do this with a using statement if the file
#     # is optionally written to but without having to buffer the output?
#     output_file = FLAGS.predict_output_file
#     if output_file is not None:
#         outf = open(output_file, 'w')
#     else:
#         outf = None
#
#     if FLAGS.predict_file:
#         print('Reading sampleData to predict from' + FLAGS.predict_file)
#         predict_input = tf.gfile.GFile(FLAGS.predict_file, mode="r")
#     else:
#         print("No input file specified; reading from STDIN")
#         predict_input = sys.stdin
#
#     for i, source in enumerate(predict_input):
#         # Strip off newline
#         source = source[:-1]
#
#         predictions = classify(source)
#
#         if outf is not None:
#             outf.write(('%d\t%s\t%s\t%s\n' % (i, label, source_text, predicted)) \
#                        .encode('utf-8'))
#             # Since the model can take a while to predict, flush often
#             # so the end-user can actually see progress when writing to a file
#             if i % 10 == 0:
#                 outf.flush()
#         else:
#             print(('Instance %d\t%s\t%s' % \
#                    (i, source.encode('utf-8'), ' '.join(predictions))).encode('utf-8'))
#
#     if outf is not None:
#         outf.close()


def set_param(name, val):
    """
    Programmatically sets a variable used in FLAGS.  This method is useful for
    configuring the model if Equilid is being retrained manually via function
    call.
    """
    setattr(FLAGS, name, val)

def textToIds(text, charToId):
    """
    Segments words within a text to individual characters and
    replaces them with an Index in their vocabulary.
    :param text: text to convert
    :param charToId: dict containing key, value pairs like: 'c'->'5'
    :return: sequence of numbers representing each character in the text
    """
    charList = list("".join(text.strip().split()))
    charToIds = []
    for ch in charList:
        charToIds.append(int(charToId[ch]))
    return charToIds

if __name__== "__main__":
    char_vocab_path = FLAGS.data_dir + '/vocab_src.dict'
    lang_vocab_path = FLAGS.data_dir + '/vocab.tgt'
    src_dict_path = FLAGS.data_dir + '/src_vocab.dict'

    char_itos, char_stoi = get_vocab(char_vocab_path)
    lang_itos, lang_stoi = get_vocab(lang_vocab_path)
    _ , src_dict = get_vocab(src_dict_path)

    print("Char Vocab itos:{} \n stoi:{}".format(char_itos, char_stoi))
    print("Lang Vocab itos:{} \n stoi:{}".format(lang_itos, lang_stoi))

    class vocab_cls(object):
        def __init__(self, itos, stoi):
            self.itos = itos
            self.stoi = stoi

    # In target file for language we only see indices of the language
    # However, the language dict here maps from string to index
    # If we pass this dict as vocab to the torchtext.bucketiterator later,
    # it  causes problems during training because bucketiterator
    # will try to lookup the language code numbers read from the target file
    # within the vocab, which won't be there.
    # Hence creating a one-to-one mapping for the indices in lang
    lang_index_stoi = {}
    for k, v in lang_stoi.items():
        lang_index_stoi[str(v)] = v
    lang_index_itos = list(lang_stoi.values())

    charSpecialsId = len(char_itos)
    langSpecialsId = len(lang_itos)

    for _, spl_symbol in _SPECIALS.items():
        char_itos.append(charSpecialsId)
        char_stoi[spl_symbol] = charSpecialsId
        lang_index_itos.append(langSpecialsId)
        lang_index_stoi[spl_symbol] = langSpecialsId
        langSpecialsId += 1
        charSpecialsId += 1

    # char_tabular = torchtext.datasets.SequenceTaggingDataset(char_vocab_path,
    #                                              fields=[('text', torchtext.data.Field(use_vocab=False)),
    #                                                      ('labels',None)])
    # lang_tabular = torchtext.datasets.SequenceTaggingDataset(lang_vocab_path,
    #                                              fields=[('text', torchtext.data.Field(use_vocab=False,
    #                                                                                    init_token=tgtField.SYM_SOS,
    #                                                                                    eos_token=tgtField.SYM_EOS))])
    #
    # print("char vocab len:{}".format(len(char_tabular)))
    # print("lang vocab len:{}".format(len(lang_tabular)))
    # # print("verifying char fields:{}".format(char_tabular.fields))
    # print("verifying lang fields:{}".format(lang_tabular.fields))
    # srcField.build_vocab(char_tabular.text, max_size=50000)
    # tgtField.build_vocab(lang_tabular.text, max_size=50000)
    # print("Source vocab len:{}".format(len(srcField.vocab.stoi)))
    # print("Target vocab itos:{} and stoi:{}".format(tgtField.vocab.itos,tgtField.vocab.stoi))

    # Create model.
    # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    tgtVocabClass = vocab_cls(lang_itos, lang_stoi)
    srcVocabClass= vocab_cls(char_itos, char_stoi)
    tgtIndexVocabClass = vocab_cls(lang_index_itos, lang_index_stoi)

    FLAGS.char_vocab_size = len(char_itos)
    FLAGS.lang_vocab_size = len(lang_index_itos)
    print("Char vocab size:{} Lang vocab size:{}".format(FLAGS.char_vocab_size, FLAGS.lang_vocab_size))

    if FLAGS.train_mode:
        train(srcVocabClass, tgtIndexVocabClass)

    srcText= "how are you"
    srcTextToIds = textToIds(srcText, src_dict)
    print(classify(srcTextToIds, tgtVocabClass.itos))
