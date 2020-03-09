import numpy as np
import hgtk
from keras.models import load_model
import han2one
from han2one import char2onehot
alp = han2one.alp
uniquealp = han2one.uniquealp

bilstm_sa_char = load_model('model/rec_self_char_b64-50-0.9826.hdf5')
bilstm_sa_char_200 = load_model('model/rec_self_char_b64_200-50-0.9853.hdf5')

sample = ['너무 아무것도 하기 싫어요','그럼 아무것도 하지 마']

def featurize_onlychar(corpus,maxlen):
    rnn_char  = np.zeros((len(corpus),1,maxlen,len(alp)))
    for i in range(len(corpus)):
        s    = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                rnn_char[i][0,-j-1,:] = char2onehot(s[-j-1])
    return rnn_char

answer_set = ['상관 없는 말이네요','의도는 비슷해요','토픽은 비슷하네요','의도와 토픽이 겹치네요','같은 말이군요!']

def return_pred(text1,text2,maxlen,model):
    input_text = [text1+' 셒 '+ text2]
    trans_text = featurize_onlychar(input_text,maxlen)
    trans_revise = [trans_text[0],np.zeros((1,64))]
    scores     = model.predict(trans_revise)[0]
    print(scores)
    print(answer_set[np.argmax(scores)])

#return_pred('오늘왜 이렇게 덥지','이렇게 오늘 더운 이유를 알 수 있을까요',200,bilstm_sa_char_200)

def return_answer(text,corpus,model):
    input_corpus = [text+' 셒 '+z for z in corpus]
    trans_corpus = featurize_onlychar(input_corpus,100)
    trans_revise = [[trans,np.zeros((1,64))] for trans in trans_corpus]
    score_list   = [model.predict(trans) for trans in trans_revise]
    scores       = [z[0][4] for z in score_list]
    answer       = corpus[np.argmax(score_list)]
    print(scores,answer)

#return_answer('왜 이렇게 암것도 하기 싫지',sample,bilstm_sa_char)

#### API functions

def fast_pred(text1,text2):
    input_text = [text1+' 셒 '+ text2]
    trans_text = featurize_onlychar(input_text,100)
    trans_revise = [trans_text[0],np.zeros((1,64))]
    scores     = bilstm_sa_char.predict(trans_revise)[0]
    return answer_set[np.argmax(scores)]

def fast_docu(text,corpus):
    input_corpus = [text+' 셒 '+z for z in corpus]
    trans_corpus = featurize_onlychar(input_corpus,100)
    trans_revise = [[trans,np.zeros((1,64))] for trans in trans_corpus]
    score_list   = [bilstm_sa_char.predict(trans) for trans in trans_revise]
    scores       = [z[0][4] for z in score_list]
    answer       = corpus[np.argmax(scores)]
    return answer


