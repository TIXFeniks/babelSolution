from itertools import islice, chain

import numpy as np
import tensorflow as tf

from vocab import Vocab
from bleu import compute_bleu
from batch_iterator import iterate_minibatches
from models.gnmt_lstm import AttentiveTranslationModel
from models.transformer_other import Model


def batch_generator(src_path, dst_path, batch_size=16, batches_per_epoch=None, skip_lines=0):
    with open(src_path) as f_src, open(dst_path) as f_dst:
        while True:
            num_lines = batches_per_epoch * batch_size if not batches_per_epoch is None else None

            f_src = islice(f_src, skip_lines, num_lines)
            f_dst = islice(f_dst, skip_lines, num_lines)

            batch = []

            for src_line, dst_line in zip(f_src, f_dst):
                if len(src_line) == 0 or len(dst_line) == 0: continue

                batch.append([src_line[:-1], dst_line[:-1]])

                if len(batch) >= batch_size:
                    yield (batch)
                    batch = []

            if batches_per_epoch is not None:
                raise StopIteration('File is read till the end, but too few batches were extracted')


def batch_generator_over_dataset(src, dst, batch_size=16, batches_per_epoch=None):
    for batch in iterate_minibatches(list(zip(src, dst)), batchsize=batch_size, shuffle=True):
        batch = batch[0] # Removing strange np.array wrapper
        batch_src = [pair[0] for pair in batch]
        batch_dst = [pair[1] for pair in batch]

        yield (batch_src, batch_dst)


def compute_bleu_for_model(model, sess, inp_voc, out_voc, src_val, dst_val, model_type):
    src_val_ix = inp_voc.tokenize_many(src_val)

    inp = tf.placeholder(tf.int32, [None, None])
    translations = []

    if model_type == 'gnmt':
        sy_translations = model.symbolic_translate(inp, greedy=True)[0]
    elif model_type == 'transformer':
        sy_translations = model.symbolic_translate(inp, mode='greedy', max_len=100).best_out
    else:
        raise NotImplemented("Unknown model")

    for batch in iterate_minibatches(src_val_ix, batchsize=64):
        translations += sess.run([sy_translations], feed_dict={inp: batch[0]})[0].tolist()

    outputs = out_voc.detokenize_many(translations, unbpe=True)
    targets = Vocab.remove_bpe_many(dst_val)
    references = [[t] for t in targets]

    bleu = compute_bleu(references, outputs)[0]

    return bleu


def should_stop_early(val_scores, use_last_n=5):
    """Determines if the model does not improve by the validation scores"""
    if len(val_scores) < use_last_n: return False

    return np.argmax(val_scores[-5:]) is 0


def create_model(name, inp_voc, out_voc, hp):
    if name == 'gnmt':
        return AttentiveTranslationModel(name, inp_voc, out_voc, hp.get('emb_size', 64),
                                         hp.get('hid_size', 64), hp.get('attn_size', 64))
    elif name == 'transformer':
        return Model(name, inp_voc, out_voc, **hp)
    else:
        raise ValueError('Model "{}" is unkown'.format(name))
