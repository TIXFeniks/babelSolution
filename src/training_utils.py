from itertools import islice, chain

import numpy as np
import tensorflow as tf

from vocab import Vocab
from bleu import compute_bleu
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


def compute_bleu_for_model(model, sess, inp_voc, out_voc, src_val, dst_val, model_name, config, max_len=200):
    src_val_ix = inp_voc.tokenize_many(src_val)

    inp = tf.placeholder(tf.int32, [None, None])
    translations = []

    if model_name == 'gnmt':
        sy_translations = model.symbolic_translate(inp, greedy=True)[0]
    elif model_name == 'transformer':
        sy_translations = model.symbolic_translate(inp, mode='greedy', max_len=max_len,
                                                   back_prop=False, swap_memory=True).best_out
    else:
        raise NotImplemented("Unknown model")

    for batch in iterate_minibatches(src_val_ix, batchsize=config.get('batch_size_for_inference', 64)):

        translations += sess.run([sy_translations], feed_dict={inp: batch[0][:, :max_len]})[0].tolist()

    outputs = out_voc.detokenize_many(translations, unbpe=True, deprocess=True)
    outputs = [out.split() for out in outputs]

    targets = out_voc.remove_bpe(dst_val)
    targets = [[t.split()] for t in targets]

    bleu = compute_bleu(targets, outputs)[0]

    return bleu


def should_stop_early(val_scores, use_last_n=5):
    """Determines if the model does not improve by the validation scores"""
    if len(val_scores) < use_last_n: return False

    # Stop early if we did not see a good BLEU for a long time
    return val_scores[-use_last_n] >= max(val_scores[-use_last_n + 1:])


def create_model(name, inp_voc, out_voc, hp):
    if name == 'gnmt':
        return AttentiveTranslationModel(name, inp_voc, out_voc, hp.get('emb_size', 64),
                                         hp.get('hid_size', 64), hp.get('attn_size', 64))
    elif name == 'transformer':
        return Model(name, inp_voc, out_voc, **hp)
    else:
        raise ValueError('Model "{}" is unkown'.format(name))


def create_gpu_options(config):
    gpu_options = tf.GPUOptions(allow_growth=True)

    if config.get('gpu_memory_fraction'):
        gpu_options.per_process_gpu_memory_fraction = config.get('gpu_memory_fraction', 0.95)

    return gpu_options


def create_optimizer(hp):
    lr = hp.get('lr', 1e-4)
    beta2 = hp.get('beta2', 0.98)

    return tf.train.AdamOptimizer(learning_rate=lr, beta2=beta2)


def iterate_minibatches(*to_split, **kwargs):
    """
        generates batches from the data, passed as first arguments

        arguments:
        -batchsize: the number of rows in a batch
        -shuffle: if True shuffles the data and yields shuffled batches
    """
    batchsize = kwargs.get('batchsize', 1)
    shuffle = kwargs.get('shuffle', False)

    res = [np.array(x) for x in to_split]
    size = res[0].shape[0]

    for x in res:
        assert x.shape[0] == size

    if shuffle:
        indices = np.arange(size)
        np.random.shuffle(indices)

    for start_idx in range(0, size, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [x[excerpt] for x in res]
