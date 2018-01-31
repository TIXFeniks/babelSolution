import numpy as np
import tensorflow as tf
from functools import lru_cache
from lib.smx import sliced_argmax
from lib.utils import nested_map, is_namedtuple
from lib.inference import GreedyInference, BeamSearchInference


class TranslateModel:

    def __init__(self, name, inp_voc, out_voc, **hp):
        """ Each model must have name, vocabularies and a hyperparameter dict """
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp

    def encode(self, batch, **flags):
        """
        Encodes symbolic input and returns initial state of decode
        :param batch: {
            inp: int32 matrix [batch,time] or whatever your model can encode
            inp_len: int vector [batch_size]
        }
        --------------------------------------------------
        :returns: dec_state, nested structure of tensors, batch-major
        """
        raise NotImplementedError()

    def decode(self, dec_state, words, **flags):
        """
        Performs decode step on given words.

        dec_state: nested structure of tensors, batch-major
        words: int vector [batch_size]
        ------------------------------------------------------
        :returns: new_dec_state, nested structure of tensors, batch-major
        """
        raise NotImplementedError()

    def get_logits(self, state, **flags):
        raise NotImplementedError()

    def sample(self, dec_state, base_scores, slices, k, sampling_strategy='greedy', sampling_temperature=None, **flags):
        """
        Samples top-K new words for each hypothesis from a beam.
        Decoder states and base_scores of hypotheses for different inputs are concatenated like this:
            [x0_hypo0, x0_hypo1, ..., x0_hypoN, x1_hypo0, ..., x1_hypoN, ..., xM_hypoN

        :param dec_state: nested structure of tensors, batch-major
        :param base_scores: [batch_size], log-probabilities of hypotheses in dec_state with additive penalties applied
        :param slices: start indices of each input
        :param k: [], int, how many hypotheses to sample per input
        :returns: best_hypos, words, scores,
            best_hypos: in-beam hypothesis index for each sampled token, [batch_size / slice_size, k], int
            words: new tokens for each hypo, [batch_size / slice_size, k], int
            scores: log P(words | best_hypos), [batch_size / slice_size, k], float32
        """
        logits = self.get_logits(dec_state)

        if sampling_temperature is not None:
            logits /= sampling_temperature

        n_hypos, voc_size = tf.shape(logits)[0], tf.shape(logits)[1]
        batch_size = tf.shape(slices)[0]

        if sampling_strategy == 'greedy':
            logp = tf.nn.log_softmax(logits, 1) + base_scores[:, None]
            best_scores, best_indices = sliced_argmax(logp, slices, k)
            best_hypos = tf.where(tf.not_equal(best_indices, -1),
                                  tf.floordiv(best_indices, voc_size) + slices[:, None],
                                  tf.fill(tf.shape(best_indices), -1))
            best_words = tf.where(tf.not_equal(best_indices, -1),
                                  tf.mod(best_indices, voc_size),
                                  tf.fill(tf.shape(best_indices), -1))

            # compute delta scores. If best_hypos == -1, best_scores == -inf, best_hypos are 0 to avoid IndexError
            best_delta_scores = best_scores - tf.gather(base_scores, tf.maximum(0, best_hypos))

        elif sampling_strategy == 'random':
            logp = tf.nn.log_softmax(logits, 1)

            best_hypos = tf.range(0, n_hypos)[:, None]

            best_words = tf.cast(tf.multinomial(logp, k), tf.int32)
            best_words_flat = (tf.range(0, batch_size) * voc_size)[:, None] + best_words

            best_delta_scores = tf.gather(tf.reshape(logp, [-1]), best_words_flat)
        else:
            raise ValueError("sampling_strategy must be in ['random','greedy']")

        return (best_hypos, best_words, best_delta_scores)

    def get_rdo(self, dec_state):
        if hasattr(dec_state, 'rdo'):
            return dec_state.rdo
        raise NotImplementedError()

    def get_attnP(self, dec_state):
        """
        Returns attnP

        dec_state: [..., batch_size, ...]
        ---------------------------------
        Ret: attnP
            attnP: [batch_size, ninp]
        """
        if hasattr(dec_state, 'attnP'):
            return dec_state.attnP
        raise NotImplementedError()

    def shuffle(self, dec_state, flat_indices):
        """
        Selects hypotheses from model decoder state by given indices.
        :param dec_state: a nested structure of tensors representing model state
        :param flat_indices: int32 vector of indices to select
        :returns: dec state elements for given flat_indices only
        """
        return nested_map(lambda var: tf.gather(var, flat_indices, axis=0), dec_state)

    def switch(self, condition, state_on_true, state_on_false):
        """
        Composes a new stack.best_dec_state out of new dec state when new_is_better and old dec state otherwise
        :param condition: a boolean condition vector of shape [batch_size]
        """
        return nested_map(lambda x, y: tf.where(condition, x, y), state_on_true, state_on_false)

    def symbolic_translate(self, inp, mode='beam_search', **flags):
        """
        A function that takes a symbolic input tokens and returns symolic translations
        :param inp: input tokens, int32[batch, time]
        :param mode: 'greedy', 'sample', or 'beam_search'
        :param flags: anything else you want to pass to decoder, encode, decode, sample, etc.
        :return: a class with .best_out, .best_scores containing symbolic tensors for translations
        """
        batch_placeholder = {'inp': inp}
        assert mode in ('greedy', 'sample', 'beam_search', 'beam_search_old')

        # create default flags
        beam_search_flags_in_hp = ['beam_size', 'beam_spread', 'len_alpha', 'attn_beta']
        for flag in beam_search_flags_in_hp:
            if flag in self.hp and flag not in flags:
                flags[flag] = self.hp[flag]
        flags['sampling_strategy'] = 'sample' if mode == 'sample' else 'greedy'

        if mode in ('greedy', 'sample'):
            return GreedyInference(
                model=self,
                batch_placeholder=batch_placeholder,
                sampling_strategy='best' if mode == 'greedy' else 'sample',
                force_bos=self.hp.get('force_bos', False),
                **flags)

        elif mode == 'beam_search':
            return BeamSearchInference(
                model=self,
                batch_placeholder=batch_placeholder,
                force_bos=self.hp.get('force_bos', False),
                **flags
            )
