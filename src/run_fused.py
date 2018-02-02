import os
import json
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import ewma

from vocab import Vocab
from src.training_utils import *
from lib.tensor_utils import infer_mask, initialize_uninitialized_variables

from models.transformer_ensemble import Ensemble
from models.transformer_fused import Model
from models.transformer_lm import TransformerLM

def run_model(model_name, config):
    """Loads model and runs it on data"""

    input_path = config.get('input_path')
    output_path = config.get('output_path')

    src_data = open(input_path, 'r', encoding='utf-8').read().splitlines()

    inp_voc = Vocab.from_file('{}/1.voc'.format(config.get('data_path')))
    out_voc = Vocab.from_file('{}/2.voc'.format(config.get('data_path')))

    # We get paths to trained models via this argument
    # We do not save optimizer state, so we can read all files from dir
    paths_to_models = ['{}/{}'.format(config.get('models_dir'), m) for m in os.listdir(config.get('models_dir'))]
    print('Found models to ensemble:', paths_to_models)

    hp = json.load(open(config.get('hp_file_path'), 'r', encoding='utf-8')) if config.get('hp_file_path') else {}
    gpu_options = create_gpu_options(config)
    max_len = config.get('max_input_len', 200)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        lm = TransformerLM('lm2', out_voc, **hp)
        if config.get('target_lm_path'):
            lm_weights = np.load(config.get('target_lm_path'))
            ops = []

            for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, lm.name):
                if w.name in lm_weights:
                    ops.append(tf.assign(w, lm_weights[w.name]))
                else:
                    print(w.name, 'not initialized')

            sess.run(ops);
        else:
            raise ValueError("Must specify LM path!")

        models = []
        assigns = []
        print('Loading models')
        for i, model_path in enumerate(paths_to_models):
            print('Loading model from', model_path)
            # Loading weights is not an easy task:
            # They were saved in transformer/ scope,
            # but now we should rename them into transformer_i/ to avoid collision
            curr_model_name = 'transformer_' + str(i)
            curr_model = Model(curr_model_name, inp_voc, out_voc, lm, **hp)
            models.append(curr_model)

            curr_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, curr_model_name)

            # Loading current model state
            for key, value in np.load(model_path).items():
                desired_key = curr_model_name + '/' + '/'.join(key.split('/')[1:]).replace('transformer/', '')
                print('Renaming {} => {}'.format(key, desired_key))

                # weight_var = tf.get_variable(desired_key)
                weight_var = None
                for v in curr_weights:
                    if v.name == desired_key:
                        weight_var = v
                        break

                assert weight_var != None
                assigns.append(tf.assign(weight_var, value))

        sess.run(assigns)

        ensemble = Ensemble(model_name, models, inp_voc, out_voc, lm, **hp)
        initialize_uninitialized_variables(sess)

        print('Generating translations')
        inp = tf.placeholder(tf.int32, [None, None])
        sy_translations = ensemble.symbolic_translate(inp, back_prop=False, swap_memory=True).best_out
        translations = []

        for batch in tqdm(iterate_minibatches(src_data, batchsize=config.get('batch_size_for_inference'))):
            batch_data_ix = inp_voc.tokenize_many(batch[0])[:, :max_len]
            trans_ix = sess.run([sy_translations], feed_dict={inp: batch_data_ix})[0]
            # parameter "deprocess=True" gets rid of BOS and EOS
            trans = out_voc.detokenize_many(trans_ix, unbpe=True, deprocess=True)
            translations.extend(trans)

        print('Saving the results into %s' % output_path)
        with open(output_path, 'wb') as output_file:
            output_file.write('\n'.join(translations).encode('utf-8'))

def main():
    parser = argparse.ArgumentParser(description='Run project commands')

    parser.add_argument('model')
    parser.add_argument('--data_path')
    parser.add_argument('--models_dir')
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--target_lm_path')
    parser.add_argument('--hp_file_path')
    parser.add_argument('--batch_size_for_inference', type=int)
    parser.add_argument('--max_input_len', type=int)
    parser.add_argument('--gpu_memory_fraction', type=float)

    args = parser.parse_args()

    config = vars(args)
    config = dict(filter(lambda x: x[1], config.items()))  # Getting rid of None vals

    print('Running %s model!' % args.model)
    print('Config', config)
    run_model(args.model, config)


if __name__ == '__main__':
    main()
