from lib.layers import *
from lib.tensor_utils import *
from lib.transformer_layers import TransformerEncoder, TransformerDecoder


class Transformer:
    """ The main model, consisting of encoder, decoder and logits """
    def __init__(self, name, inp_voc, out_voc, *_args,
                 hid_size=256, **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp
        hp['force_bos'] = hp.get('force_bos', True)

        with tf.variable_scope(name):
            self.enc = TransformerEncoder("enc", inp_voc, hid_size=hid_size, **hp)
            self.dec = TransformerDecoder("dec", inp_voc, hid_size=hid_size,
                                          allow_lookahead=False,
                                          **hp)

            projection_matrix = None
            if hp.get('dwwt', False):
                projection_matrix = tf.transpose(self.dec.emb_out.mat)

            self.logits = Dense('logits', hid_size, out_voc.size(),
                                activaction=nop, W=projection_matrix)

    def symbolic_score(self, inp, out, is_train=False):
        """ Computes a sequence of logits aka model prediction. Used for training """

        # rdo: [batch_size * nout * hid_dim]
        enc_out, enc_attn_mask = self.enc(inp, is_train=is_train)

        dec_out, dec_attn_mask = self.dec(enc_out, out, enc_attn_mask, is_train=is_train)

        return self.logits(dec_out)

    def symbolic_translate(self, inp, out, is_train=False):
        raise NotImplementedError("TODO(jheuristic) finish merging")
