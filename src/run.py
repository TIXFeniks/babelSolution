import os
import json
import argparse
# import subprocess

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import ewma

from bleu import compute_bleu
from models.gnmt_lstm import AttentiveTranslationModel
from vocab import Vocab
from src.training_utils import batch_generator_over_dataset, compute_bleu_for_model
from lib.tensor_utils import infer_mask, initialize_uninitialized_variables
from batch_iterator import iterate_minibatches

MODEL_NAME = 'super_gnmt_model'

def run_gnmt(config):
    """
    Trains GNMT model, saves its weights and optimizer state
    """

    input_path = config.get('input_path')
    output_path = config.get('output_path')

    src_data = open(input_path, 'r', encoding='utf-8').read().splitlines()

    inp_voc = Vocab.from_file('{}/1.voc'.format(config.get('data_path')))
    out_voc = Vocab.from_file('{}/2.voc'.format(config.get('data_path')))

    # Hyperparameters
    hp = json.load(open(config.get('hp_file'), 'r', encoding='utf-8')) if config.get('hp_file') else {}
    emb_size = hp.get('emb_size', 32)
    hid_size = hp.get('hid_size', 32)
    attn_size = hp.get('attn_size', 32)
    lr = hp.get('lr', 1e-4)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.get('gpu_memory_fraction', 0.3))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = AttentiveTranslationModel(MODEL_NAME, inp_voc, out_voc, emb_size, hid_size, attn_size)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, MODEL_NAME)

        # Loading model state
        w_values = np.load(config.get('model_path'))
        curr_var_names = set(w.name for w in weights)
        var_names_in_file = set(w_values.keys())

        assert curr_var_names == var_names_in_file

        assigns = [tf.assign(var, tf.constant(w_values[var.name])) for var in weights]
        sess.run(assigns)

        print('Running the model')
        src_data_ix = inp_voc.tokenize_many(src_data)
        inp = tf.placeholder(tf.int32, [None, None])
        sy_translations = model.symbolic_translate(inp, greedy=True)[0]

        translations = []

        for batch in iterate_minibatches(src_data_ix, batchsize=config.get('run_batch_size')):
            translations += sess.run([sy_translations], feed_dict={inp: batch[0]})[0].tolist()

        translations = [t[1:] for t in translations] # Removing BOS

        outputs = out_voc.detokenize_many(translations, unbpe=True)

        print('Saveing the results into %s' % output_path)
        with open(output_path, 'wb') as output_file:
            output_file.write('\n'.join(outputs).encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='model')

    model_parser = subparsers.add_parser('gnmt')

    model_parser.add_argument('--data_path')
    model_parser.add_argument('--model_path')
    model_parser.add_argument('--input_path')
    model_parser.add_argument('--output_path')
    model_parser.add_argument('--hp_file')
    model_parser.add_argument('--run_batch_size', type=int)
    model_parser.add_argument('--gpu_memory_fraction', type=float)

    args = parser.parse_args()

    if args.model == 'gnmt':
        print('Running gnmt!')

        config = vars(args)
        config = dict(filter(lambda x: x[1], config.items())) # Getting rid of None vals

        run_gnmt(config)
    else:
        print('No model name is provided')

if __name__ == '__main__':
    main()
