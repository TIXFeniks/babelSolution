from lib.layers import *
from lib.tensor_utils import *
from lib.transformer_layers import TransformerEncoder, FeedforwardBlock
import tensorflow as tf

class TransformerDiscriminator:
    """ Discriminator, using Transformer's encoder """
    def __init__(self, name, voc, *_args, num_post_layers = 1, hid_size=256, res_steps='nlda', dropout = 0, **hp):
        self.name = name
        self.voc = voc
        self.dropout = dropout
        self.hp = hp
        self.num_post_layers = num_post_layers
        hp['force_bos'] = hp.get('force_bos', True)

        with tf.variable_scope(name):
            self.enc = TransformerEncoder("enc", voc, hid_size=hid_size, res_steps=res_steps, **hp)
            self.post = [ResidualLayerWrapper(
                'post-ff-%i' % i,
                FeedforwardBlock(
                    'post-ff-%i' % i,
                    inp_size=hid_size,
                    hid_size=hid_size,
                    out_size=hid_size,
                    relu_dropout=0),
                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(self.num_post_layers)]
            self.logit = Dense('logit', hid_size,1, activation=nop)
    def predict(self, inp, is_train=False):
        """ Computes a model prediction."""
        encoded, enc_mask = self.enc(inp, is_train=is_train)

        for layer in range(self.num_post_layers):
            #no attention so far
            #enc_inp = self.post[layer](enc_inp, attn_mask)
            encoded = self.post[layer](encoded)
        logits = self.logit(stupidmax(encoded, mask = enc_mask[:,0,0,:], axis=1))
        return logits#tf.nn.log_softmax(logits)
