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


from models.transformer_fused import Model
from models.transformer_lm import TransformerLM

def run_model(model_name, config):
    """Loads model and runs it on data"""

    input_path = config.get('input_path')
    output_path = config.get('output_path')

    src_data = open(input_path, 'r', encoding='utf-8').read().splitlines()

    inp_voc = Vocab.from_file('{}/1.voc'.format(config.get('data_path')))
    out_voc = Vocab.from_file('{}/2.voc'.format(config.get('data_path')))

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
        model = Model(model_name, inp_voc, out_voc, lm, **hp)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

        # Loading model state
        w_values = np.load(config.get('model_path'))
        curr_var_names = set(w.name for w in weights)
        #var_names_in_file = set(w_values.keys())  # I AM A FIX

        #assert curr_var_names == var_names_in_file

        assigns = [tf.assign(var, tf.constant(w_values[var.name])) for var in weights]
        sess.run(assigns)

        initialize_uninitialized_variables(sess)

        print('Generating translations')
        inp = tf.placeholder(tf.int32, [None, None])

        assert model_name != 'gnmt', 'gnmt no longer supported'
        sy_translations = model.symbolic_translate(inp, back_prop=False, swap_memory=True).best_out

        translations = []

        for batch in tqdm(iterate_minibatches(src_data, batchsize=config.get('batch_size_for_inference'))):

            try:
                batch_data_ix = inp_voc.tokenize_many(batch[0])[:, :max_len]
                trans_ix = sess.run([sy_translations], feed_dict={inp: batch_data_ix})[0]
                # deprocess = True gets rid of BOS and EOS
                trans = out_voc.detokenize_many(trans_ix, unbpe=True, deprocess=True)
            except Exception as e:
                print('Ahtung! Batch is failed:', e)
                # we failed this batch. At least one sample is broken
                trans = []
                src_rows = batch[0]
                for row in src_rows:
                    try:
                        row_ix = inp_voc.tokenize_many([row])[:, :max_len]  # [1, inp_len]

                        row_trans_ix = sess.run([sy_translations],
                                                feed_dict={inp: row_ix})[0]  # [1, out_len]

                        # deprocess = True gets rid of BOS and EOS
                        row_trans = out_voc.detokenize_many(row_trans_ix,
                                                            unbpe=True, deprocess=True)  # [1]
                        trans.append(row_trans[0])
                    except Exception as e:
                        print('Ahtung! We could not even translate a sample:', e)
                        # we failed this very row. Use src as fallback
                        trans.append(str(row).replace('\n', ''))  # cast to str just in case

            translations.extend(trans)

        print('Saving the results into %s' % output_path)

        try:
            with open(output_path, 'wb') as output_file:
                output_file.write('\n'.join(translations).encode('utf-8'))
        except Exception as e:
            print('Failed conversion:', e)
            # we failed at conversion somewhere. Write one row at a time
            with open(output_path, 'w') as output_file:
                for translation in translations:
                    try:
                        output_file.write(translation.encode('utf-8'))
                    except:
                        output_file.write(" - failed writing this line to file , . !")

                    output_file.write("\n")
def main():
    parser = argparse.ArgumentParser(description='Run project commands')

    parser.add_argument('model')
    parser.add_argument('--data_path')
    parser.add_argument('--model_path')
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
