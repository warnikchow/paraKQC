import numpy as np
import sys

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

para = read_data('data/paraKQC_v1.txt')

import itertools

def make_para(tuplet):
    res   = []
    pairs = [comb for comb in itertools.combinations(list(range(len(tuplet))), 2)]
    for i in range(len(pairs)):
        line = []
        line.append(4)
        line.append(tuplet[pairs[i][0]])
        line.append(tuplet[pairs[i][1]])
        res.append(line)
    return res

def make_something(tuplet1,tuplet2,i1,i2,type):
    utt1 = tuplet1[i2%10][2]
    utt2 = tuplet2[i1%10][2]
    line = []
    line.append(type)
    line.append(utt1)
    line.append(utt2)
    return line

# divide the corpus into 10-tuplets
# finally aims to make up 4 classes of sentence pairs
# 4. paraphrase (maximum 45 * 1000)
# 3. same topic and same intention, but not paraphrase
# 2. same topic but different intention
# 1. same intention but different topic
# 0. different intention and different topic
tuples = []
for i in range(1000):
    start = i*10
    end   = (i+1)*10
    tuple = para[start:end]
    tuples.append(tuple)

para_total     = []
for i in range(len(tuples)):
    tuplet     = [z[2] for z in tuples[i]]
    para_set   = make_para(tuplet)
    para_total = para_total+para_set

non_para_total = []
tuple_pairs = [comb for comb in itertools.combinations(list(range(len(tuples))), 2)]
for i in range(len(tuple_pairs)):
    if i%10000==0:
        print(i)
    type   = 0
    tuple1 = tuples[tuple_pairs[i][0]]
    tuple2 = tuples[tuple_pairs[i][1]]
    topic  = tuple1[0][0]==tuple2[0][0]
    intent = tuple1[0][1]==tuple2[0][1]
    if topic and intent:
        type = 3
    if topic and not intent:
        type = 2
    if not topic and intent:
        type = 1
    non_para_set = make_something(tuple1,tuple2,tuple_pairs[i][0],tuple_pairs[i][1],type)
    non_para_total.append(non_para_set)

data_total = para_total + non_para_total

def find_int_to4(x):
    count_all = [[],[],[],[],[]]
    for i in range(len(x)):
        count_all[x[i][0]].append(i)
    return count_all

count_all = find_int_to4(data_total)
'''
from random import shuffle
for i in range(len(count_all)):
    shuffle(count_all[i])

np.save('shuffled',count_all)
'''
count_all = np.load('shuffled.npy')
count_all = np.load('shuffled.npy',allow_pickle=True)

numbers = []
for i in range(len(count_all)):
    numbers.append(len(count_all[i])*0.9)

train_index = []
test_index  = []
for i in range(len(count_all)):
    train_index = train_index + count_all[i][:int(numbers[i])]
    test_index  = test_index  + count_all[i][int(numbers[i]):]

import fasttext
model_ft = fasttext.load_model('vectors/model_drama.bin')

import hgtk
import han2one
from han2one import shin_onehot, cho_onehot, char2onehot
alp = han2one.alp
uniquealp = han2one.uniquealp

####### Two methods adopted from
####### https://github.com/warnikchow/kcharemb

def featurize_rnnchar(corpus,index_list,wdim,maxlen):
    rnn_char  = np.zeros((len(corpus),maxlen,len(alp)))
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%10000 ==0:
            print(i)
        ind  = index_list[i]
        s    = corpus[ind][1] + ' 셒 ' + corpus[ind][2]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                rnn_char[i][-j-1,:] = char2onehot(s[-j-1])
                if s[-j-1] in model_ft:
                    rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
    return rnn_char, rnn_total

total_index = train_index + test_index
total_rec_char, total_rec_dense = featurize_rnnchar(data_total,total_index,100,100)
total_rec_char_200, total_rec_dense_200 = featurize_rnnchar(data_total,total_index,100,200)
total_label = [data_total[z][0] for z in total_index]

####### ' 셒 ' adopted as a separation token for Korean
####### Very rarely used in the corpus

def featurize_rnnchar_halfsep(corpus,index_list,maxlen):
    rnn_char  = np.zeros((len(corpus),2*maxlen+3,len(alp)))
    for i in range(len(corpus)):
        if i%10000 ==0:
            print(i)
        ind  = index_list[i]
        s1    = corpus[ind][1]
        s2    = ' 셒 ' + corpus[ind][2]
        for j in range(len(s1)):
            if j < maxlen and hgtk.checker.is_hangul(s1[-j-1])==True:
                rnn_char[i][100-j-1,:] = char2onehot(s1[-j-1])
        for j in range(len(s2)):
            if j < maxlen and hgtk.checker.is_hangul(s2[j])==True:
                rnn_char[i][100+j,:]   = char2onehot(s2[j])
    return rnn_char

total_index = train_index + test_index
total_rec_char_halfsep = featurize_rnnchar_halfsep(data_total,total_index,100)
total_label = [data_total[z][0] for z in total_index]

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(total_label), total_label)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
from keras.models import Sequential
import keras.layers as layers
from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
from keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback
from sklearn import metrics
class Metricsf1macro(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(val_predict,axis=1)
        val_targ = self.validation_data[1]
        _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
        _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
        _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
        _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
        _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
        _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_f1s_w.append(_val_f1_w)
        self.val_recalls_w.append(_val_recall_w)
        self.val_precisions_w.append(_val_precision_w)
        print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
        print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro = Metricsf1macro()

class Metricsf1macro_forself(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
  self.val_f1s_w = []
  self.val_recalls_w = []
  self.val_precisions_w = []
 def on_epoch_end(self, epoch, logs={}):
  if len(self.validation_data)>2:
   val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[2]
  else:
   val_predict = np.asarray(self.model.predict(self.validation_data[0]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  self.val_f1s_w.append(_val_f1_w)
  self.val_recalls_w.append(_val_recall_w)
  self.val_precisions_w.append(_val_precision_w)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_self = Metricsf1macro_forself()

####### Construct CNN model ## Todo

####### Construct BiLSTM & BiLSTM-SA models

from keras.layers import LSTM
from keras.layers import Bidirectional

def validate_bilstm(result,y,hidden_lstm,hidden_dim,cw,val_sp,bat_size,filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(len(result[0]), len(result[0][0]))))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(result,y,validation_split=val_sp,epochs=50,batch_size=bat_size,callbacks=callbacks_list,class_weight=cw)

#validate_bilstm(total_rec_dense,total_label,32,128,class_weights,0.1,64,'model/rec_dense_b64')
#validate_bilstm(total_rec_char,total_label,32,128,class_weights,0.1,64,'model/rec_char_b64')
validate_bilstm(total_rec_char_halfsep,total_label,32,256,class_weights,0.1,64,'model_lrec/rec_char_b64_256')

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, TimeDistributed
import keras.backend as K
from keras.layers.core import Dropout

def validate_rnn_self_drop(x_rnn,x_y,hidden_lstm,hidden_con,hidden_dim,cw,val_sp,bat_size,filename):
    char_r_input = Input(shape=(len(x_rnn[0]),len(x_rnn[0][0])),dtype='float32')
    r_seq = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(char_r_input)
    r_att = Dense(hidden_con, activation='tanh')(r_seq)
    att_source   = np.zeros((len(x_rnn),hidden_con))
    att_test     = np.zeros((len(x_rnn),hidden_con))
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec      = Dense(hidden_con,activation='relu')(att_input)
    att_vec      = Dropout(0.3)(att_vec)
    att_vec      = Dense(hidden_con,activation='relu')(att_vec)
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,r_att])
    att_vec = Dense(len(x_rnn[0]),activation='softmax')(att_vec)
    att_vec = layers.Reshape((len(x_rnn[0]),1))(att_vec)
    r_seq   = layers.multiply([att_vec,r_seq])
    r_seq   = Lambda(lambda x: K.sum(x, axis=1))(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    main_output = Dense(int(max(x_y)+1),activation='softmax')(r_seq)
    model = Sequential()
    model = Model(inputs=[char_r_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_self,checkpoint]
    model.fit([x_rnn,att_source],x_y,validation_split=val_sp,epochs=50,batch_size= bat_size ,callbacks=callbacks_list,class_weight=cw)

#validate_rnn_self_drop(total_rec_dense,total_label,32,64,256,class_weights,0.1,64,'model/rec_self_dense_b64')
#validate_rnn_self_drop(total_rec_char,total_label,32,64,256,class_weights,0.1,64,'model3.6/rec_self_char_b64')
#validate_rnn_self_drop(total_rec_char_200,total_label,32,64,256,class_weights,0.1,64,'model3.6/rec_self_char_b64_200')
#validate_rnn_self_drop(total_rec_char_halfsep,total_label,32,64,256,class_weights,0.1,64,'model3.6/rec_self_char_b64_halfsep')
validate_rnn_self_drop(total_rec_char_halfsep,total_label,32,64,256,class_weights,0.1,64,'model_lrec/rec_self_char_b64_halfsep')

def validate_rnn_self_drop_for36(x_rnn,x_y,hidden_lstm,hidden_con,hidden_dim,cw,val_sp,bat_size,filename):
    char_r_input = Input(shape=(len(x_rnn[0]),len(x_rnn[0][0])),dtype='float32')
    r_seq = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(char_r_input)
    r_att = Dense(hidden_con, activation='tanh')(r_seq)
    att_source   = np.zeros((len(x_rnn),hidden_con))
    att_test     = np.zeros((len(x_rnn),hidden_con))
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec      = Dense(hidden_con,activation='relu')(att_input)
    att_vec      = Dropout(0.3)(att_vec)
    att_vec      = Dense(hidden_con,activation='relu')(att_vec)
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,r_att])
    att_vec = Dense(len(x_rnn[0]),activation='softmax')(att_vec)
    att_vec = layers.Reshape((len(x_rnn[0]),1))(att_vec)
    r_seq   = layers.multiply([att_vec,r_seq])
    r_seq   = Lambda(lambda x: K.sum(x, axis=1))(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    main_output = Dense(int(max(x_y)+1),activation='softmax')(r_seq)
    model = Sequential()
    model = Model(inputs=[char_r_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_self,checkpoint]
    model.fit([x_rnn,att_source],x_y,validation_split=val_sp,epochs=50,batch_size= bat_size ,callbacks=callbacks_list,class_weight=cw)

validate_rnn_self_drop_for36(total_rec_char_200,total_label,32,64,256,class_weights,0.1,64,'model3.6/rec_self_char_b64_200')
validate_rnn_self_drop_for36(total_rec_char_halfsep,total_label,32,64,256,class_weights,0.1,64,'model3.6/rec_self_char_b64_halfsep')

####### Parallel BiLSTM

from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Bidirectional, Concatenate
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Lambda
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

class Metricsf1macro_4input(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
  self.val_f1s_w = []
  self.val_recalls_w = []
  self.val_precisions_w = []
 def on_epoch_end(self, epoch, logs={}):
  if len(self.validation_data)>2:
   val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1],self.validation_data[2],self.validation_data[3]]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[4]
  else:
   val_predict = np.asarray(self.model.predict(self.validation_data[0]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  self.val_f1s_w.append(_val_f1_w)
  self.val_recalls_w.append(_val_recall_w)
  self.val_precisions_w.append(_val_precision_w)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_4input = Metricsf1macro_4input()

def featurize_rnnchar_parallel(corpus,index_list,maxlen):
    rnn_char1  = np.zeros((len(corpus),maxlen,len(alp)))
    rnn_char2  = np.zeros((len(corpus),maxlen,len(alp)))
    for i in range(len(corpus)):
        if i%10000 ==0:
            print(i)
        ind  = index_list[i]
        s1    = corpus[ind][1]
        s2    = corpus[ind][2]
        for j in range(len(s1)):
            if j < maxlen and hgtk.checker.is_hangul(s1[-j-1])==True:
                rnn_char1[i][100-j-1,:] = char2onehot(s1[-j-1])
        for j in range(len(s2)):
            if j < maxlen and hgtk.checker.is_hangul(s2[-j-1])==True:
                rnn_char2[i][100-j-1,:]   = char2onehot(s2[-j-1])
    return rnn_char1, rnn_char2

total_rec_char1, total_rec_char2 = featurize_rnnchar_parallel(data_total,total_index,100)

####### The code adopted from https://github.com/warnikchow/coaudiotext
####### Speech and Text denotes 1st and 2nd document respectively here

def validate_speech_self_text_self(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM-SA
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_layer = Bidirectional(LSTM(hidden_lstm_speech,return_sequences=True))(speech_input)
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source= np.zeros((len(rnn_speech),hidden_con))
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')
    speech_att_vec   = Dense(hidden_con, activation='relu')(speech_att_input)
    speech_att_vec   = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    speech_att_vec   = Dense(len(rnn_speech[0]),activation='softmax')(speech_att_vec)
    speech_att_vec   = layers.Reshape((len(rnn_speech[0]),1))(speech_att_vec)
    speech_output= layers.multiply([speech_att_vec,speech_layer])
    speech_output= Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output= Dense(hidden_dim, activation='relu')(speech_output)
    ##### Text BiLSTM-SA
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att   = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')
    text_att_vec    = Dense(hidden_con,activation='relu')(text_att_input)
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    text_att_vec = Dense(len(rnn_text[0]),activation='softmax')(text_att_vec)
    text_att_vec = layers.Reshape((len(rnn_text[0]),1))(text_att_vec)
    text_output  = layers.multiply([text_att_vec,text_layer])
    text_output  = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Total output
    output    = layers.concatenate([speech_output, text_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=50,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)

validate_speech_self_text_self(total_rec_char1,total_rec_char2,total_label,32,64,32,256,class_weights,0.1,64,'model_lrec/rec_parallel_b64')

####### Cross-attention
####### The code adopted from https://github.com/warnikchow/coaudiotext
####### Speech and Text denotes 1st and 2nd document respectively here

def validate_rnn_self_text_self_cross(rnn_speech,rnn_text,train_y,hidden_lstm_speech,hidden_con,hidden_lstm_text,hidden_dim,cw,val_sp,bat_size,filename):
    ##### Speech BiLSTM-SA
    speech_input = Input(shape=(len(rnn_speech[0]),len(rnn_speech[0][0])), dtype='float32')
    speech_layer = Bidirectional(LSTM(hidden_lstm_speech,return_sequences=True))(speech_input)
    speech_att   = Dense(hidden_con, activation='tanh')(speech_layer)
    speech_att_source= np.zeros((len(rnn_speech),hidden_con))
    speech_att_input = Input(shape=(hidden_con,),dtype='float32')
    speech_att_vec   = Dense(hidden_con, activation='relu')(speech_att_input)
    speech_att_vec   = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([speech_att_vec,speech_att])
    ##### Text BiLSTM-SA
    text_input = Input(shape=(len(rnn_text[0]),len(rnn_text[0][0])),dtype='float32')
    text_layer = Bidirectional(LSTM(hidden_lstm_text,return_sequences=True))(text_input)
    text_att = Dense(hidden_con, activation='tanh')(text_layer)
    text_att_source = np.zeros((len(rnn_text),hidden_con))
    text_att_input  = Input(shape=(hidden_con,), dtype='float32')
    text_att_vec    = Dense(hidden_con,activation='relu')(text_att_input)
    text_att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([text_att_vec,text_att])
    #####
    speech_att_vec   = Dense(hidden_con,activation='softmax')(speech_att_vec)
    text_att_vec     = Dense(hidden_con,activation='softmax')(text_att_vec)
    #att_vec          = layers.concatenate([speech_att_vec, text_att_vec])
    cross_speech_att_vec   = Dense(len(rnn_speech[0]),activation='softmax')(text_att_vec)
    cross_text_att_vec     = Dense(len(rnn_text[0]),activation='softmax')(speech_att_vec)
    #####
    cross_speech_att_vec   = layers.Reshape((len(rnn_speech[0]),1))(cross_speech_att_vec)
    speech_output    = layers.multiply([cross_speech_att_vec,speech_layer])
    speech_output    = Lambda(lambda x: K.sum(x, axis=1))(speech_output)
    speech_output    = Dense(hidden_dim, activation='relu')(speech_output)
    #####
    cross_text_att_vec     = layers.Reshape((len(rnn_text[0]),1))(cross_text_att_vec)
    text_output   = layers.multiply([cross_text_att_vec,text_layer])
    text_output   = Lambda(lambda x: K.sum(x, axis=1))(text_output)
    text_output  = Dense(hidden_dim, activation='relu')(text_output)
    ##### Total output
    output    = layers.concatenate([speech_output, text_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(int(max(train_y)+1),activation='softmax')(output)
    model = Sequential()
    #####
    model = Model(inputs=[speech_input,speech_att_input,text_input,text_att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    #####
    callbacks_list = [metricsf1macro_4input,checkpoint]
    model.summary()
    #####
    model.fit([rnn_speech,speech_att_source,rnn_text,text_att_source],train_y,validation_split = val_sp,epochs=50,batch_size= bat_size,callbacks=callbacks_list,class_weight=cw)

validate_rnn_self_text_self_cross(total_rec_char1,total_rec_char2,total_label,32,64,32,256,class_weights,0.1,64,'model_lrec/rec_ca_b64')
