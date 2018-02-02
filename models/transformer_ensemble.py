#!/usr/bin/env python3
import math
import numpy as np
import tensorflow as tf
from lib.layers import *
from lib.tensor_utils import *
from collections import namedtuple
from . import TranslateModel


class Ensemble(TranslateModel):


    def __init__(self, name, models, inp_voc, out_voc, lm, **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp
        self.lm = lm
        self.debug = hp.get('debug', None)

        self.models = models # Here we keep our models
        self.DecState = namedtuple("transformer_state", ['state_of_model_%i'%i for i in range(len(models))])

    def encode(self, batch, **kwargs):
        states = [m.encode(batch, **kwargs) for m in self.models]

        return self.DecState(*states)

    def decode(self, states, words=None, is_train=False, **kwargs):
        states = [m.decode(s, words, is_train, **kwargs) for s, m in zip(states, self.models)]

        return self.DecState(*states)

    def get_rdo(self, dec_states, **kwargs):
        dec_state = dec_states[0]
        return dec_state.rdo, dec_state.out_seq

    def get_attnP(self, dec_states, **kwargs):
        dec_state = dec_states[0]
        return dec_state.attnP

    def get_logits(self, dec_states, **flags):

        logits = []
        for state, model in zip(dec_states, self.models):
            logits.append(model.get_logits(state, **flags))

        return sum(logits) / len(self.models)
