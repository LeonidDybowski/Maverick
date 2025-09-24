"""
This code was taken from:

Danzi, M.C., Dohrn, M.F., Fazal, S. et al. 
Deep structured learning for variant prioritization in Mendelian diseases. 
Nat Commun 14, 4167 (2023). https://doi.org/10.1038/s41467-023-39306-7

Modifications:
- Adapted to run with Horovod for distributed training
- Tensorflow lifted from 2.7 to 2.17 for compatibility
"""

import scipy
import sys, getopt, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from official.nlp.modeling.layers import PositionEmbedding, TransformerEncoderBlock
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from transformers import TFT5EncoderModel, T5Tokenizer
from transformers import set_seed
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    roc_auc_score
)
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import categorical_crossentropy
from transformers import TFT5EncoderModel, T5Tokenizer,T5Config
from sklearn.utils import resample
from scipy.stats import rankdata
from datetime import datetime

# Set float32 policy
mixed_precision.set_global_policy('float32')
pd.options.mode.chained_assignment = None

set_seed(57)
np.random.seed(57)
tf.random.set_seed(57)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, dataFrameIn, tokenizer, T5Model,
                 batch_size=32, padding=100, n_channels_emb=1024, n_channels_mm=51,
                 n_classes=3, shuffle=True, returnStyle=1, **kwargs):
        super().__init__(**kwargs)
        self.returnStyle = returnStyle
        self.padding = padding
        self.dim = self.padding + self.padding + 1
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels_emb = n_channels_emb
        self.n_channels_mm = n_channels_mm
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataFrameIn = dataFrameIn  # Expecting DataFrame with reset index!
        self.tokenizer = tokenizer
        self.T5Model = T5Model

        self.shard_ids = self.list_IDs
        self.num_samples = len(self.shard_ids)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.shard_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.shard_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.shard_ids[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        thisBatchSize = len(list_IDs_temp)
        altEmbeddings = np.zeros((thisBatchSize, self.dim, self.n_channels_emb))
        mm_alt = np.zeros((thisBatchSize, self.dim, self.n_channels_mm))
        mm_orig = np.zeros((thisBatchSize, self.dim, self.n_channels_mm))
        nonSeq = np.zeros((thisBatchSize, 12))
        y = np.empty((thisBatchSize), dtype=int)
        AMINO_ACIDS = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,
                       'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,
                       'V':17,'W':18,'Y':19}
        T5AltSeqTokens = []

        for i, ID in enumerate(list_IDs_temp):
            # Use iloc for integer indexing
            transcriptID = self.dataFrameIn.iloc[ID]['TranscriptID']
            changePos = self.dataFrameIn.iloc[ID]['ChangePos'] - 1
            if changePos < 0:
                changePos = 0
            AltSeq = self.dataFrameIn.iloc[ID]['AltSeq']
            if AltSeq[-1] != "*":
                AltSeq = AltSeq + "*"
            seqLenAlt = len(AltSeq) - 1
            startPos = 0
            if changePos > self.padding:
                if (changePos + self.padding) < seqLenAlt:
                    startPos = changePos - self.padding
                elif seqLenAlt >= self.dim:
                    startPos = seqLenAlt - self.dim
            endPos = changePos + self.padding
            if changePos < self.padding:
                if self.dim < seqLenAlt:
                    endPos = self.dim
                else:
                    endPos = seqLenAlt
            T5AltSeqTokens.append(" ".join(AltSeq[startPos:endPos]))

            WTSeq = self.dataFrameIn.iloc[ID]['WildtypeSeq']
            if WTSeq[-1] != "*":
                WTSeq = WTSeq + "*"
            seqLen = len(WTSeq) - 1
            startPos = 0
            if changePos > self.padding:
                if (changePos + self.padding) < seqLen:
                    startPos = int(changePos - self.padding)
                elif seqLen >= self.dim:
                    startPos = int(seqLen - self.dim)
            endPos = int(changePos + self.padding)
            if changePos < self.padding:
                if self.dim < seqLen:
                    endPos = int(self.dim)
                else:
                    endPos = int(seqLen)
            T5AltSeqTokens.append(" ".join(WTSeq[startPos:endPos]))

            tmp = np.load(f"../HHMProfiles/{transcriptID}_MMSeqsProfile.npz", allow_pickle=True)
            tmp = tmp['arr_0']
            seqLen = tmp.shape[0]
            startPos = changePos - self.padding
            endPos = changePos + self.padding + 1
            startOffset = 0
            endOffset = self.dim
            if changePos < self.padding:
                startPos = 0
                startOffset = self.padding - changePos
            if (changePos + self.padding) >= seqLen:
                endPos = seqLen
                endOffset = self.padding + seqLen - changePos
            mm_orig[i, startOffset:endOffset, :] = tmp[startPos:endPos, :]

            varType = self.dataFrameIn.iloc[ID]['varType']
            WTSeq = self.dataFrameIn.iloc[ID]['WildtypeSeq']
            if varType == 'nonsynonymous SNV':
                if changePos == 0:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    altEncoded[:, 0:20] = 0
                    altEncoded[:, 50] = 0
                else:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    altEncoded[changePos, AMINO_ACIDS[WTSeq[changePos]]] = 0
                    altEncoded[changePos, AMINO_ACIDS[AltSeq[changePos]]] = 1
            elif varType == 'stopgain':
                if changePos == 0:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    altEncoded[:, 0:20] = 0
                    altEncoded[:, 50] = 0
                elif seqLenAlt > seqLen:
                    altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLen):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    for j in range(seqLen, seqLenAlt):
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLen:, 50] = 1
                else:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    altEncoded[changePos:, 0:20] = 0
                    altEncoded[changePos:, 50] = 0
            elif varType == 'stoploss':
                altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                altEncoded[:seqLen, :] = tmp
                for j in range(seqLen, seqLenAlt):
                    altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                altEncoded[seqLen:, 50] = 1
            elif varType == 'synonymous SNV':
                altEncoded = tmp
            elif ((varType == 'frameshift deletion') | (varType == 'frameshift insertion') | (varType == 'frameshift substitution')):
                if seqLen < seqLenAlt:
                    altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLen):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    for j in range(seqLen, seqLenAlt):
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLen:, 50] = 1
                elif seqLen > seqLenAlt:
                    for j in range(changePos, seqLenAlt):
                        tmp[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        tmp[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    for j in range(seqLenAlt, seqLen):
                        tmp[j, AMINO_ACIDS[WTSeq[j]]] = 0
                    altEncoded = tmp
                elif seqLen == seqLenAlt:
                    for j in range(changePos, seqLen):
                        tmp[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        tmp[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded = tmp
                else:
                    print('Error: seqLen comparisons did not work')
                    exit()
            elif varType == 'nonframeshift deletion':
                altNucLen = 0
                if self.dataFrameIn.loc[ID, 'alt'] != '-':
                    altNucLen = len(self.dataFrameIn.loc[ID, 'alt'])
                refNucLen = len(self.dataFrameIn.loc[ID, 'ref'])
                numAADel = int((refNucLen - altNucLen) / 3)
                if (seqLen - numAADel) == seqLenAlt:
                    for j in range(changePos, (changePos + numAADel)):
                        tmp[j, :20] = 0
                    altEncoded = tmp
                elif seqLen >= seqLenAlt:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLenAlt):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLenAlt:, 0:20] = 0
                    altEncoded[seqLenAlt:, 50] = 0
                elif seqLen < seqLenAlt:
                    altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLen):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLen:, 0:20] = 0
                    altEncoded[seqLen:, 50] = 0
                else:
                    print('Error: seqLen comparisons did not work for nonframeshift deletion')
                    exit()
            elif varType == 'nonframeshift insertion':
                refNucLen = 0
                if self.dataFrameIn.loc[ID, 'ref'] != '-':
                    refNucLen = len(self.dataFrameIn.loc[ID, 'ref'])
                altNucLen = len(self.dataFrameIn.loc[ID, 'alt'])
                numAAIns = int((altNucLen - refNucLen) / 3)
                if (seqLen + numAAIns) == seqLenAlt:
                    altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                    altEncoded[:changePos, :] = tmp[:changePos, :]
                    altEncoded[(changePos + numAAIns):, :] = tmp[changePos:, :]
                    for j in range(numAAIns):
                        altEncoded[(changePos + j), AMINO_ACIDS[AltSeq[(changePos + j)]]] = 1
                    altEncoded[:, 50] = 1
                elif seqLen < seqLenAlt:
                    altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLen):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    for j in range(seqLen, seqLenAlt):
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLen:, 50] = 1
                elif seqLen >= seqLenAlt:
                    altEncoded = np.zeros((seqLen, self.n_channels_mm))
                    altEncoded[:seqLen, :] = tmp
                    for j in range(changePos, seqLenAlt):
                        altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                        altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                    altEncoded[seqLenAlt:, 0:20] = 0
                    altEncoded[seqLenAlt:, 50] = 0
                else:
                    print('Error: seqLen comparisons did not work for nonframeshift insertion')
                    exit()
            elif varType == 'nonframeshift substitution':
                refNucLen = len(self.dataFrameIn.loc[ID, 'ref'])
                altNucLen = len(self.dataFrameIn.loc[ID, 'alt'])
                if refNucLen > altNucLen:
                    if seqLen > seqLenAlt:
                        numAADel = int((refNucLen - altNucLen) / 3)
                        if (seqLen - numAADel) == seqLenAlt:
                            for j in range(changePos, (changePos + numAADel)):
                                tmp[j, :20] = 0
                            altEncoded = tmp
                        else:
                            altEncoded = np.zeros((seqLen, self.n_channels_mm))
                            altEncoded[:seqLen, :] = tmp
                            for j in range(changePos, seqLenAlt):
                                altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                                altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                            altEncoded[seqLenAlt:, 0:20] = 0
                            altEncoded[seqLenAlt:, 50] = 0
                    elif seqLen < seqLenAlt:
                        altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLen):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        for j in range(seqLen, seqLenAlt):
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        altEncoded[seqLen:, 50] = 1
                    else:
                        altEncoded = np.zeros((seqLen, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLen):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                elif refNucLen < altNucLen:
                    if seqLen < seqLenAlt:
                        numAAIns = int((altNucLen - refNucLen) / 3)
                        if (seqLen + numAAIns) == seqLenAlt:
                            altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                            altEncoded[:changePos, :] = tmp[:changePos, :]
                            altEncoded[(changePos + numAAIns):, :] = tmp[changePos:, :]
                            for j in range(numAAIns):
                                altEncoded[(changePos + j), AMINO_ACIDS[AltSeq[(changePos + j)]]] = 1
                            altEncoded[:, 50] = 1
                        else:
                            altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                            altEncoded[:seqLen, :] = tmp
                            for j in range(changePos, seqLen):
                                altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                                altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                            for j in range(seqLen, seqLenAlt):
                                altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                            altEncoded[:, 50] = 1
                    elif seqLen > seqLenAlt:
                        altEncoded = np.zeros((seqLen, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLenAlt):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        altEncoded[seqLenAlt:, 0:20] = 0
                        altEncoded[seqLenAlt:, 50] = 0
                    else:
                        altEncoded = np.zeros((seqLen, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLen):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                elif refNucLen == altNucLen:
                    if seqLen == seqLenAlt:
                        altEncoded = np.zeros((seqLen, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        altEncoded[changePos, AMINO_ACIDS[WTSeq[changePos]]] = 0
                        altEncoded[changePos, AMINO_ACIDS[AltSeq[changePos]]] = 1
                    elif seqLen > seqLenAlt:
                        altEncoded = np.zeros((seqLen, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLenAlt):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        altEncoded[seqLenAlt:, 0:20] = 0
                        altEncoded[seqLenAlt:, 50] = 0
                    elif seqLen < seqLenAlt:
                        altEncoded = np.zeros((seqLenAlt, self.n_channels_mm))
                        altEncoded[:seqLen, :] = tmp
                        for j in range(changePos, seqLen):
                            altEncoded[j, AMINO_ACIDS[WTSeq[j]]] = 0
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        for j in range(seqLen, seqLenAlt):
                            altEncoded[j, AMINO_ACIDS[AltSeq[j]]] = 1
                        altEncoded[seqLen:, 50] = 1
                    else:
                        print('non-frameshift substitution comparisons failed')
                        exit()
                else:
                    print('Error: nonframeshift substitution nucleotide length comparison did not work')
                    exit()
            startPos = changePos - self.padding
            endPos = changePos + self.padding + 1
            startOffset = 0
            endOffset = self.dim
            if changePos < self.padding:
                startPos = 0
                startOffset = self.padding - changePos
            if (changePos + self.padding) >= seqLenAlt:
                endPos = seqLenAlt
                endOffset = self.padding + seqLenAlt - changePos
            mm_alt[i, startOffset:endOffset, :] = altEncoded[startPos:endPos, :]

            nonSeq[i] = self.dataFrameIn.iloc[ID][['controls_AF', 'controls_nhomalt', 'pLI', 'pNull', 'pRec', 'mis_z', 'lof_z', 'CCR', 'GDI', 'pext', 'RVIS_ExAC_0.05', 'gerp']]
            y[i] = self.labels[ID]

        allTokens = self.tokenizer.batch_encode_plus(T5AltSeqTokens, add_special_tokens=True, padding=True, return_tensors="tf")
        input_ids = allTokens['input_ids'][::2]
        attnMask = allTokens['attention_mask'][::2]
        embeddings = self.T5Model(input_ids, attention_mask=attnMask)
        allEmbeddings = np.asarray(embeddings.last_hidden_state)
        for i in range(thisBatchSize):
            seq_len = (np.asarray(attnMask)[i] == 1).sum()
            seq_emb = allEmbeddings[i][1:seq_len-1]
            altEmbeddings[i, :seq_emb.shape[0], :] = seq_emb
        
        if self.returnStyle == 1:
            X = {'mm_orig_seq': mm_orig, 'mm_alt_seq': mm_alt, 'non_seq_info': nonSeq}
        else:
            X = {'alt_cons': mm_alt, 'alt_emb': altEmbeddings, 'non_seq_info': nonSeq}
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def MaverickArchitecture1(input_shape=201, classes=3, classifier_activation='softmax', **kwargs):
    input0 = tf.keras.layers.Input(shape=(input_shape, 51), name='mm_orig_seq')
    input1 = tf.keras.layers.Input(shape=(input_shape, 51), name='mm_alt_seq')
    input2 = tf.keras.layers.Input(shape=12, name='non_seq_info')

    # Project inputs
    x_orig = tf.keras.layers.EinsumDense('...x,xy->...y', output_shape=64, bias_axes='y')(input0)
    x_alt = tf.keras.layers.EinsumDense('...x,xy->...y', output_shape=64, bias_axes='y')(input1)

    posEnc_wt = PositionEmbedding(max_length=input_shape)(x_orig)
    x_orig = tf.keras.layers.Masking()(x_orig)
    x_orig = tf.keras.layers.Add()([x_orig, posEnc_wt])
    x_orig = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)(x_orig)
    x_orig = tf.keras.layers.Dropout(0.05)(x_orig)

    posEnc_alt = PositionEmbedding(max_length=input_shape)(x_alt)
    x_alt = tf.keras.layers.Masking()(x_alt)
    x_alt = tf.keras.layers.Add()([x_alt, posEnc_alt])
    x_alt = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)(x_alt)
    x_alt = tf.keras.layers.Dropout(0.05)(x_alt)
    
    for _ in range(6):
        transformer = TransformerEncoderBlock(
            16, 256, tf.keras.activations.relu, output_dropout=0.1, attention_dropout=0.1)
        x_orig = transformer(x_orig)
        x_alt = transformer(x_alt)

    first_token_tensor_orig = tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(x_orig)
    x_orig = tf.keras.layers.Dense(units=64, activation='tanh')(first_token_tensor_orig)
    x_orig = tf.keras.layers.Dropout(0.05)(x_orig)

    first_token_tensor_alt = tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(x_alt)
    x_alt = tf.keras.layers.Dense(units=64, activation='tanh')(first_token_tensor_alt)
    x_alt = tf.keras.layers.Dropout(0.05)(x_alt)

    diff = tf.keras.layers.Subtract()([x_alt, x_orig])
    combined = tf.keras.layers.concatenate([x_alt, diff])

    input2Dense1 = tf.keras.layers.Dense(64, activation='relu')(input2)
    input2Dense1 = tf.keras.layers.Dropout(0.05)(input2Dense1)
    x = tf.keras.layers.concatenate([combined, input2Dense1])
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='output')(x)
    model = tf.keras.Model(inputs=[input0, input1, input2], outputs=x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.85)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def MaverickArchitecture2(input_shape=201, embeddingSize=1024, mmSize=51, classes=3, classifier_activation='softmax', **kwargs):
    input0 = tf.keras.layers.Input(shape=(input_shape, mmSize), name='alt_cons')
    input1 = tf.keras.layers.Input(shape=(input_shape, embeddingSize), name='alt_emb')
    input2 = tf.keras.layers.Input(shape=12, name='non_seq_info')

    # Project to lower-dimensional embedding
    alt_cons = tf.keras.layers.EinsumDense('...x,xy->...y', output_shape=64, bias_axes='y')(input0)

    posEnc_alt = PositionEmbedding(max_length=input_shape)(alt_cons)
    alt_cons = tf.keras.layers.Masking()(alt_cons)
    alt_cons = tf.keras.layers.Add()([alt_cons, posEnc_alt])
    alt_cons = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(alt_cons)
    alt_cons = tf.keras.layers.Dropout(0.05)(alt_cons)

    for _ in range(6):
        transformer = TransformerEncoderBlock(
            16, 256, tf.keras.activations.relu, output_dropout=0.1, attention_dropout=0.1)
        alt_cons = transformer(alt_cons)

    first_token_tensor_alt = tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(alt_cons)
    alt_cons = tf.keras.layers.Dense(units=64, activation='tanh')(first_token_tensor_alt)
    alt_cons = tf.keras.layers.Dropout(0.05)(alt_cons)

    #alt_emb = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.5))(input1)
    forward = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32, dropout=0.5), return_sequences=False)
    backward = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32, dropout=0.5), return_sequences=False, go_backwards=True)
    alt_emb = tf.keras.layers.Bidirectional(forward, backward_layer=backward)(input1)
    
    alt_emb = tf.keras.layers.Dropout(0.2)(alt_emb)

    structured = tf.keras.layers.Dense(64, activation='relu')(input2)
    structured = tf.keras.layers.Dropout(0.05)(structured)

    x = tf.keras.layers.concatenate([alt_cons, alt_emb, structured])
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='output')(x)

    model = tf.keras.Model(inputs=[input0, input1, input2], outputs=x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.85)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

training_set_path = 'trainingSet_data2025_ann2025_hg38_test_infevers.txt'
validation_set_path = 'validationSet_data2025_ann2025_hg38_test_infevers.txt'
test_set_path = 'testSetInfevers.txt'

tokenizer = T5Tokenizer.from_pretrained("../prot_t5_xl_bfd", do_lower_case=False, local_files_only=True)
T5Model = TFT5EncoderModel.from_pretrained("../prot_t5_xl_bfd", local_files_only=True)

batchSize = 64

trainingData = pd. read_csv(training_set_path, sep='\t', low_memory=False)
trainingData = trainingData.reset_index(drop=True)  # FIX: Reset index to ensure integer-based indexing
trainingData.loc[trainingData['GDI'] > 2000, 'GDI'] = 2000
trainingDataNonSeqInfo = trainingData[['controls_AF', 'controls_nhomalt', 'pLI', 'pNull', 'pRec', 'mis_z', 'lof_z', 'CCR', 'GDI', 'pext', 'RVIS_ExAC_0.05', 'gerp']].copy(deep=True)
trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_AF'].isna(), 'controls_AF'] = 0
trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_nhomalt'].isna(), 'controls_nhomalt'] = 0
trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_nhomalt'] > 10, 'controls_nhomalt'] = 10
trainingDataNonSeqMedians = trainingDataNonSeqInfo.median()
trainingDataNonSeqInfo = trainingDataNonSeqInfo.fillna(trainingDataNonSeqMedians)
trainingDataNonSeqInfo = np.asarray(trainingDataNonSeqInfo.to_numpy()).astype(np.float32)

qt = QuantileTransformer(subsample=10**6, random_state=0, output_distribution='uniform')
qt = qt.fit(trainingDataNonSeqInfo)
trainingDataNonSeqInfo = qt.transform(trainingDataNonSeqInfo)
trainingData.loc[:, ['controls_AF', 'controls_nhomalt', 'pLI', 'pNull', 'pRec', 'mis_z', 'lof_z', 'CCR', 'GDI', 'pext', 'RVIS_ExAC_0.05', 'gerp']] = trainingDataNonSeqInfo

# Pad the dataframe so that its length is equally divisible by the batch size
remainder = len(trainingData) % batchSize
if remainder != 0:
    padding = trainingData.iloc[-(batchSize - remainder):].copy()
    trainingData = pd.concat([trainingData, padding], ignore_index=True)
    print('Length of training set after padding: ', len(trainingData))

def preprocess_dataframe(df):
    df = df.reset_index(drop=True)
    df.loc[df['GDI'] > 2000, 'GDI'] = 2000
    dfNonSeqInfo = df[['controls_AF', 'controls_nhomalt', 'pLI', 'pNull', 'pRec', 'mis_z', 'lof_z', 'CCR', 'GDI', 'pext', 'RVIS_ExAC_0.05', 'gerp']].copy(deep=True)
    dfNonSeqInfo.loc[dfNonSeqInfo['controls_AF'].isna(), 'controls_AF'] = 0
    dfNonSeqInfo.loc[dfNonSeqInfo['controls_nhomalt'].isna(), 'controls_nhomalt'] = 0
    dfNonSeqInfo.loc[dfNonSeqInfo['controls_nhomalt'] > 10, 'controls_nhomalt'] = 10
    dfNonSeqInfo = dfNonSeqInfo.fillna(trainingDataNonSeqMedians)
    dfNonSeqInfo = np.asarray(dfNonSeqInfo.to_numpy()).astype(np.float32)
    dfNonSeqInfo = qt.transform(dfNonSeqInfo)
    df.loc[:, ['controls_AF', 'controls_nhomalt', 'pLI', 'pNull', 'pRec', 'mis_z', 'lof_z', 'CCR', 'GDI', 'pext', 'RVIS_ExAC_0.05', 'gerp']] = dfNonSeqInfo
    return df

validationData = pd.read_csv(validation_set_path, sep='\t', low_memory=False)
testData = pd.read_csv(test_set_path, sep='\t', low_memory=False)

validationData = preprocess_dataframe(validationData)
testData = preprocess_dataframe(testData)

y_train = trainingData.loc[:, 'classLabel'].to_numpy()
y_valid = validationData.loc[:, 'classLabel'].to_numpy()
y_test = testData.loc[:, 'classLabel'].to_numpy()

encoder = LabelEncoder()
encoder.fit(trainingData.loc[:, 'classLabel'])
y_train_encoded = encoder.transform(trainingData.loc[:, 'classLabel'])
y_valid_encoded = encoder.transform(validationData.loc[:, 'classLabel'])
y_test_encoded = encoder.transform(testData.loc[:, 'classLabel'])

y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded).astype(int)
y_valid_encoded = tf.keras.utils.to_categorical(y_valid_encoded).astype(int)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded).astype(int)

'''
# Starting training Architecture 1
batchSize = 64
numEpochs = 20

training_generator1 = DataGenerator(np.arange(len(trainingData)), y_train, dataFrameIn=trainingData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=True, returnStyle=1)
validation_generator1 = DataGenerator(np.arange(len(validationData)), y_valid, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=1)
known_generator1 = DataGenerator(np.arange(len(knownData)), y_known, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=1)
novel_generator1 = DataGenerator(np.arange(len(novelData)), y_novel, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=1)

if hvd.rank() == 0 and not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
    
if hvd.rank() == 0:
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Train Architecture 1, model 1
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture1_Model1.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture1_Model1.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc1Model1 = MaverickArchitecture1()
if hvd.rank() == 0:
    Arc1Model1.summary()

history11 = Arc1Model1.fit(training_generator1, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator1)
if hvd.rank() == 0:
    Arc1Model1.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc1Model1


# Train Architecture 1, model 2
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture1_Model2.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture1_Model2.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc1Model2 = MaverickArchitecture1()
if hvd.rank() == 0:
    Arc1Model2.summary()

history12 = Arc1Model2.fit(training_generator1, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator1, class_weight={0:1,1:2,2:7})
if hvd.rank() == 0:
    Arc1Model2.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc1Model2

    
# Train Architecture 1, model 3
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture1_Model3.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture1_Model3.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc1Model3 = MaverickArchitecture1()
if hvd.rank() == 0:
    Arc1Model3.summary()

history13 = Arc1Model3.fit(training_generator1, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator1, class_weight={0:1,1:2,2:7})
if hvd.rank() == 0:
    Arc1Model3.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc1Model3


# Starting training Architecture 2
batchSize = 16
numEpochs = 20

training_generator2 = DataGenerator(np.arange(len(trainingData)), y_train, dataFrameIn=trainingData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=True, returnStyle=2)
validation_generator2 = DataGenerator(np.arange(len(validationData)), y_valid, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=2)
known_generator2 = DataGenerator(np.arange(len(knownData)), y_known, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=2)
novel_generator2 = DataGenerator(np.arange(len(novelData)), y_novel, dataFrameIn=validationData, tokenizer=tokenizer, T5Model=T5Model, batch_size=batchSize, shuffle=False, returnStyle=2)

# Train Architecture 2, model 1
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture2_Model1.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture2_Model1.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc2Model1 = MaverickArchitecture2()
if hvd.rank() == 0:
    Arc2Model1.summary()

history21 = Arc2Model1.fit(training_generator2, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator2)
if hvd.rank() == 0:
    Arc2Model1.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc2Model1


# Train Architecture 2, model 2
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture2_Model2.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture2_Model2.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc2Model2 = MaverickArchitecture2()
if hvd.rank() == 0:
    Arc2Model2.summary()

history22 = Arc2Model2.fit(training_generator2, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator2)
if hvd.rank() == 0:
    Arc2Model2.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc2Model2


# Train Architecture 2, model 3
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture2_Model3.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture2_Model3.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc2Model3 = MaverickArchitecture2()
if hvd.rank() == 0:
    Arc2Model3.summary()

history23 = Arc2Model3.fit(training_generator2, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator2)
if hvd.rank() == 0:
    Arc2Model3.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc2Model3


# Train Architecture 2, model 4
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture2_Model4.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture2_Model4.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc2Model4 = MaverickArchitecture2()
if hvd.rank() == 0:
    Arc2Model4.summary()

history24 = Arc2Model4.fit(training_generator2, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator2, class_weight={0:1,1:2,2:7})
if hvd.rank() == 0:
    Arc2Model4.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc2Model4


# Train Architecture 2, model 5
steps = (len(trainingData) * numEpochs) / (batchSize * hvd.size())
lr_schedule = OneCycleScheduler(0.02, steps, div_factor=10.)
modelWeightsName = 'Architecture2_Model5.weights.h5'

callbacks = [
    BroadcastGlobalVariablesCallback(0),
    MetricAverageCallback(),
    lr_schedule
]

if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/Architecture2_Model5.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

Arc2Model5 = MaverickArchitecture2()
if hvd.rank() == 0:
    Arc2Model5.summary()

history25 = Arc2Model5.fit(training_generator2, epochs=numEpochs, callbacks=callbacks, validation_data=validation_generator2, class_weight={0:1,1:2,2:7})
if hvd.rank() == 0:
    Arc2Model5.save_weights(modelWeightsName)
    tf.keras.backend.clear_session()
    del Arc2Model5
'''

# ============================
# PREDICTION
# ============================
def predict_set(df, y_true):

    evaluation_generator1 = DataGenerator(
        np.arange(len(df)),
        y_true,
        dataFrameIn=df,
        tokenizer=tokenizer,
        T5Model=T5Model,
        batch_size=batchSize,
        shuffle=False,
        returnStyle=1
    )

    evaluation_generator2 = DataGenerator(
        np.arange(len(df)),
        y_true,
        dataFrameIn=df,
        tokenizer=tokenizer,
        T5Model=T5Model,
        batch_size=batchSize,
        shuffle=False,
        returnStyle=2
    )

    Arc1Model1 = MaverickArchitecture1()
    Arc1Model2 = MaverickArchitecture1()
    Arc1Model3 = MaverickArchitecture1()
    Arc2Model1 = MaverickArchitecture2()
    Arc2Model2 = MaverickArchitecture2()
    Arc2Model3 = MaverickArchitecture2()
    Arc2Model4 = MaverickArchitecture2()
    Arc2Model5 = MaverickArchitecture2()

    Arc1Model1.load_weights('Architecture1_Model1.weights.h5')
    Arc1Model2.load_weights('Architecture1_Model2.weights.h5')
    Arc1Model3.load_weights('Architecture1_Model3.weights.h5')
    Arc2Model1.load_weights('Architecture2_Model1.weights.h5')
    Arc2Model2.load_weights('Architecture2_Model2.weights.h5')
    Arc2Model3.load_weights('Architecture2_Model3.weights.h5')
    Arc2Model4.load_weights('Architecture2_Model4.weights.h5')
    Arc2Model5.load_weights('Architecture2_Model5.weights.h5')

    # Predict
    y_pred_11 = Arc1Model1.predict(evaluation_generator1)
    y_pred_12 = Arc1Model2.predict(evaluation_generator1)
    y_pred_13 = Arc1Model3.predict(evaluation_generator1)
    y_pred_21 = Arc2Model1.predict(evaluation_generator2)
    y_pred_22 = Arc2Model2.predict(evaluation_generator2)
    y_pred_23 = Arc2Model3.predict(evaluation_generator2)
    y_pred_24 = Arc2Model4.predict(evaluation_generator2)
    y_pred_25 = Arc2Model5.predict(evaluation_generator2)
    y_pred = np.mean([y_pred_11,y_pred_12,y_pred_13,y_pred_21,y_pred_22,y_pred_23,y_pred_24,y_pred_25],axis=0)

    return y_pred

y_test_pred = predict_set(testData, y_test)

# ============================
# PRINT CLASSIFICATION REPORT
# ============================
y_test_true = y_test_encoded

testData.loc[:,['Maverick_BenignScore','Maverick_DomScore','Maverick_RecScore']]=y_test_pred

testData.loc[:,'classLabelPredicted']=np.argmax(y_test_pred,axis=1)

testData.to_csv(test_set_path.split('.')[-2].split('/')[-1] + '_predicted.txt',sep='\t',index=False)

def print_classification_report(y_pred, y_true, name_prefix):
    print(f"{name_prefix} set performance")
    print(classification_report(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=['Benign', 'Dominant', 'Recessive'],
            digits=3
    ))
    return None

print_classification_report(y_test_pred, y_test_true, "Test")
