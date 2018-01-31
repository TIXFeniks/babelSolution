import tensorflow as tf
from lib.transformer_layers import TransformerEncoder
from lib.layers import Dense
from lib.tensor_utils import infer_mask, infer_length


class TransformerLM:
    def __init__(self, name, voc, **hp):
        self.voc = voc
        self.name = name
        assert hp['force_bos'] is True, "this is build with force_bos in mind"
        hp['num_layers_enc'] = hp.get('num_layers', 1)

        with tf.variable_scope(name):
            self.enc = TransformerEncoder('main',
                                          inp_voc=voc,
                                          allow_lookahead=False,
                                          **hp)
            self.logits = Dense('logits', self.enc.hid_size, len(voc))

    def __call__(self, inp, is_train=False, after_eos=False):
        mask = infer_mask(inp, self.voc.eos, dtype=tf.float32)
        inp_len = infer_length(inp, self.voc.eos)

        enc_out, _ = self.enc(inp, is_train=is_train)

        logits = self.logits(enc_out)

        # shift logits forward
        if not after_eos:
            logits = logits[:, :-1]

        logits = tf.concat([tf.one_hot(inp[:, :1], len(self.voc)) * 100, logits], axis=1)
        return logits
