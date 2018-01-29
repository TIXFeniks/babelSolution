import numpy as np
from sortedcontainers import SortedSet

unknowns = []


class Vocab:
    _default_tokens = [ "__BOS__","__EOS__", "__PAD__"]
    
    def __init__(self, voc_path= None, sentences= None):
        self.tokens = self._default_tokens + [] # don't change default tokens
        if voc_path is not None:
            self.update_tokens_with_vocab(voc_path)
        if sentences is not None:
            self.update_tokens_with_sequences(sentences)
        self.BOS = 0 # BOS should be zero to let the model generate starting with zero as an input
        self.EOS = 1
        self.PAD = 2
    
    @property
    def token2id(self):
        return {token: i for i, token in enumerate(self.tokens)}
    
    def add_token_set(self, tokens):
        tokens.difference_update(self.tokens)
        self.tokens = self.tokens + list(tokens)
    
    def update_tokens_with_vocab(self, voc_path):
        tokens = SortedSet()
        with open(voc_path,'r') as f:
            for line in f:
                token = line.split(" ")[0]
                tokens.update([token])
        self.add_token_set(tokens)
        
    def update_tokens_with_sequences(self, sentences):
        tokens = SortedSet()
        for s in sentences:
            tokens.update(s.split(' '))
            
        self.add_token_set(tokens)
    
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
    def __len__(self):
        return len(self.tokens)
    def detokenize(self, sentence):
        return " ".join([self.tokens[token] for token in sentence])
    def tokenize_many(self, lines, max_len = None):
        max_len = max_len or max(map(lambda s: len(s.split()), lines)) + 2 # 2 for bos and eos

        matrix = np.zeros((len(lines), max_len), dtype='int32') + self.PAD
        for i, seq in enumerate(lines):
            tokens = self.tokenize(seq)[:max_len] 
            matrix[i, :len(tokens)] = tokens

        return matrix
    def detokenize_many(self, sentences):
        return [self.detokenize(sent) for sent in sentences]
    