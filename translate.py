import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import sys


SRC_PATH = sys.argv[1] if len(sys.argv) > 1 else "input/input.txt.bpe"
DST_PATH = sys.argv[2] if len(sys.argv) > 2 else "output/output.txt"

raw_src = []
with open(SRC_PATH, 'r') as f:
    raw_src = f.readlines()

def fix_ends(sents):
    for i in range(len(sents)):
        sents[i] = sents[i].replace('\n','')
        
fix_ends(raw_src)
unknowns = []

class Vocab:
    def __init__(self, sentences):
        tokens = set()
        for s in sentences:
            tokens.update(s.split(' '))
        self.tokens = [ "__BOS__","__EOS__", "__PAD__"] + list(tokens)
        self.EOS = 1
        self.BOS = 0 # BOS should be zero to let the model generate starting with zero as an input
        self.PAD = 2
        self.len = len(self.tokens)
        self.token2id = {token: i for i, token in enumerate(self.tokens)}
    def tokenize(self, sentence):
        if not sentence.endswith("__EOS__"):
            sentence += " __EOS__"
        if not sentence.startswith("__BOS__"):
            sentence = "__BOS__ " + sentence
        spl = sentence.split(' ')
        global unknowns
        unknowns += list(filter(lambda token:token not in self.token2id, spl))
        spl = list(filter(lambda token:token in self.token2id, spl))
        return [self.token2id[token] for token in spl]
    def detokenize(self, sentence):
        return " ".join([self.tokens[token] for token in sentence])
    def tokenize_many(self, sentences):
        return [self.tokenize(sent) for sent in sentences]
    def detokenize_many(self, sentences):
        return [self.detokenize(sent) for sent in sentences]
    
import pickle as pkl
with open("vocabs.pkl", 'rb') as f:
    dst_voc = pkl.load(f)
    src_voc = pkl.load(f)

X = src_voc.tokenize_many(raw_src)
print(list(set(unknowns)))
MAX_LEN = 100

import numpy as np
np.random.seed(42)
from keras.preprocessing.sequence import pad_sequences

input_sequence = T.matrix('token sequencea','int32')
input_mask = T.neq(input_sequence, src_voc.PAD)

target_values = T.matrix('actual next token','int32')
target_mask = T.neq(target_values, dst_voc.PAD)

CODE_SIZE = 512

l_in = lasagne.layers.InputLayer(shape=(None, None),input_var=input_sequence)
l_mask = lasagne.layers.InputLayer(shape=(None, None),input_var=input_mask)

#encoder
l_emb = L.EmbeddingLayer(l_in, src_voc.len, 128)

l_rnn = L.LSTMLayer(l_emb, 256, nonlinearity=T.tanh, mask_input= l_mask)
l_rnn = L.concat([l_emb,l_rnn], axis=-1)
l_encoded = l_rnn = L.LSTMLayer(l_rnn, CODE_SIZE, nonlinearity=T.tanh, mask_input= l_mask)

l_trans = L.InputLayer((None, None), input_var= target_values[:,:-1])
l_trans_mask = L.InputLayer((None, None), input_var= target_mask[:,:-1])

from agentnet.agent.recurrence import Recurrence
from agentnet.memory import AttentionLayer,LSTMCell 
from agentnet.resolver import ProbabilisticResolver, GreedyResolver

class AutoLSTMCell:
    def __init__(self, input_or_inputs, num_units = None, *args, name=None, **kwargs):
        self.p_cell = L.InputLayer((None, num_units), 
                       name="previous cell state" if name == None else name + " previous cell state")
        self.p_out = L.InputLayer((None, num_units), 
                       name="previous out state" if name == None else name + " previous out state")
        self.cell, self.out = LSTMCell(self.p_cell, self.p_out, input_or_inputs, num_units, *args,name=name, **kwargs)
    def get_automatic_updates(self):
        return {self.cell: self.p_cell, self.out: self.p_out}
    
    
class TemperatureResolver(ProbabilisticResolver):
    def __init__(self, incoming, tau, **kwargs):
        self.tau = tau
        super(TemperatureResolver, self).__init__(incoming,**kwargs)
    def get_output_for(self, policy, **kwargs):
        policy = policy ** (1/self.tau)
        policy /= policy.sum()
        return super(TemperatureResolver, self).get_output_for(policy, **kwargs) 
    
class decoder_step:
    #inputs
    encoder = L.InputLayer((None, None ,CODE_SIZE), name='encoded sequence')
    encoder_mask = L.InputLayer((None, None), name='encoded sequence')
    
    inp = L.InputLayer((None,),name='current character')
    
    l_target_emb = L.EmbeddingLayer(inp, dst_voc.len, 128)
    
    #recurrent part
    
    l_rnn1 = AutoLSTMCell(l_target_emb, 128, name="lstm1")
    
    query = L.DenseLayer(l_rnn1.out, 128, nonlinearity=None)
    attn = AttentionLayer(encoder, query, 128, mask_input= encoder_mask)['attn']
    
    l_rnn = L.concat([attn, l_rnn1.out, l_target_emb])
    
    l_rnn2 = AutoLSTMCell(l_rnn, 128, name="lstm1")
    
    next_token_probas = L.DenseLayer(l_rnn2.out, dst_voc.len, nonlinearity=T.nnet.softmax)
    
    #pick next token from predicted probas
    next_token = ProbabilisticResolver(next_token_probas)
    
    tau = T.scalar("sample temperature", "float32")
    
    next_token_temperatured = TemperatureResolver(next_token_probas, tau)
    next_token_greedy = GreedyResolver(next_token_probas)
    
    auto_updates = {**l_rnn1.get_automatic_updates(),
                    **l_rnn2.get_automatic_updates()}
    
from collections import OrderedDict
n_steps = T.scalar(dtype='int32')
feedback_loop = Recurrence(
    state_variables=OrderedDict({**decoder_step.auto_updates,
                     decoder_step.next_token:decoder_step.inp}),
    tracked_outputs=[decoder_step.next_token_probas, decoder_step.next_token],
    input_nonsequences= OrderedDict({decoder_step.encoder: l_encoded, decoder_step.encoder_mask: l_mask} ),
    batch_size=input_sequence.shape[0],
    n_steps=n_steps,
    unroll_scan=False,
)

# Model weights
weights = lasagne.layers.get_all_params(feedback_loop,trainable=True)

generated_tokens = L.get_output(feedback_loop[decoder_step.next_token])

generate_sample = theano.function([input_sequence ,n_steps],generated_tokens,
                                  updates=feedback_loop.get_automatic_updates())

feedback_loop_greedy = Recurrence(
    state_variables=OrderedDict({**decoder_step.auto_updates,
                     decoder_step.next_token_greedy:decoder_step.inp}),
    tracked_outputs=[decoder_step.next_token_probas, decoder_step.next_token_greedy],
    input_nonsequences= OrderedDict({decoder_step.encoder: l_encoded, decoder_step.encoder_mask: l_mask} ),
    batch_size=input_sequence.shape[0],
    n_steps=n_steps,
    unroll_scan=False,
)


generated_tokens_greedy = L.get_output(feedback_loop_greedy[decoder_step.next_token_greedy])

generate_sample_greedy = theano.function([input_sequence ,n_steps],generated_tokens_greedy,
                                  updates=feedback_loop_greedy.get_automatic_updates())


feedback_loop_temp = Recurrence(
    state_variables=OrderedDict({**decoder_step.auto_updates,
                     decoder_step.next_token_temperatured:decoder_step.inp}),
    tracked_outputs=[decoder_step.next_token_probas, decoder_step.next_token_temperatured],
    input_nonsequences= OrderedDict({decoder_step.encoder: l_encoded, decoder_step.encoder_mask: l_mask} ),
    batch_size=input_sequence.shape[0],
    n_steps=n_steps,
    unroll_scan=False,
)

generated_tokens_temp = L.get_output(feedback_loop_temp[decoder_step.next_token_temperatured])

generate_sample_temp = theano.function([input_sequence ,n_steps, decoder_step.tau],generated_tokens_temp,
                                  updates=feedback_loop_temp.get_automatic_updates())

weights_dict = {str(i): weight for i, weight in enumerate(weights)}

with np.load("weights.npz") as f:
    for key in weights_dict:
        weights_dict[key].set_value(f[key])
        
from tqdm import tqdm

from batch_iterator import iterate_minibatches

def translate(srcs, N=MAX_LEN,t=1, greedy=False):
    srcs = [src_voc.tokenize(src) for src in srcs]
    if len(srcs) > 1:
        srcs = pad_sequences(srcs, value= src_voc.PAD, maxlen=MAX_LEN, padding="post")
    sample_ix = generate_sample_greedy(srcs, N) if greedy else generate_sample_temp(srcs, N, t)
    random_snippet = dst_voc.detokenize_many(sample_ix)
    res = []
    for sent in random_snippet:
        if sent.find("__EOS__") > 0:
            res.append(sent[:sent.find("__EOS__")].replace("@@ ", ""))
        else:
            res.append(sent.replace("@@ ", ""))
    return res


with open(DST_PATH, 'w') as f:
    for start in tqdm(range(0, len(raw_src), 1)):
        f.write(translate(raw_src[start: start+1], t=0.2)[0]+'\n') #todo batchsize