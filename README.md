# ParaKQC
Parallel dataset of Korean Questions and Commands

## Description
### Dataset generation 
*paraKQC_v1* in *data* folder contains 10,000 utterances, namely 1,000 sets of 10 similar sentences. The corpus generation process is depicted in *mkdata.py*, to make up the whole corpus of size 545,000 utterances with the followings as subcorpus:
- Sentence similarity corpus (494,500)
- Paraphrase corpus (45,000)
### Training process
Also described in *mkdata.py*, utilizing BiLSTM, Self-attentive BiLSTM, Parallel BiLSTM, and BiLSTM Cross-attention. The demo file adopts BiLSTM-SA.

## Demo
### Requirements
- Model trained in Python 3.5 (so does the demo require)
- *git clone https://github.com/warnikchow/paraKQC* and *pip install -r Requirements*
### Usage
In python console, 
```properties
# The function that finds the most similar sentence among the candidates
>>> from pred import fast_docu

# Document of candidates
>>> t1 = '너 몇 살이냐'
>>> t2 = '거기 가는데 얼마나 걸려'
>>> t3 = '내일 다섯 시까지 옥상으로 와'
>>> t4 = '굳이 그렇게까지 해야돼'
>>> t5 = '동작 그만 밑장빼기냐'
>>> cand = [t1,t2,t3,t4,t5]

# Not the best, but relatively accurate answer
>>> fast_docu('하던 거 멈춰',cand)
'굳이 그렇게까지 해야돼'
```
