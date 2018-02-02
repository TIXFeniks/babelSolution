import os
import json
import argparse
from time import time
from datetime import timedelta
from random import shuffle

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import ewma

from vocab import Vocab
from src.training_utils import *
from lib.tensor_utils import infer_mask, initialize_uninitialized_variables, all_shapes_equal

from models.transformer_fused import Model
from models.transformer_lm import TransformerLM


def train_model(model_name, config):
    """Trains model, saves its weights and optimizer state"""

    model_path = 'trained_models/{}'.format(model_name)
    if not os.path.isdir('trained_models'): os.mkdir('trained_models')
    if not os.path.isdir(model_path): os.mkdir(model_path)

    lang = config.get('lang', 2)

    src_train_path = '{}/bpe_corpus{}.txt'.format(config.get('data_path'), lang)

    src_train = open(src_train_path, 'r', encoding='utf-8').read().splitlines()

    voc = Vocab.from_file('{}/{}.voc'.format(config.get('data_path'), lang))

    max_len = config.get('max_len', 200)

    # Hyperparameters
    hp = json.load(open(config.get('hp_file_path'), 'r', encoding='utf-8')) if config.get('hp_file_path') else {}
    gpu_options = create_gpu_options(config)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        lm = TransformerLM(model_name, voc, **hp)

        inp = tf.placeholder(tf.int32, [None, None])

        logits = lm(inp, is_train=True)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inp)

        loss = nll * infer_mask(inp, voc.eos, dtype=tf.float32)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, lm.name)
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        non_trainable_vars = list(set(all_vars).difference(set(weights)))

        grads = tf.gradients(loss, weights)
        grads = tf.clip_by_global_norm(grads, 100)[0]
        optimizer = create_optimizer(hp)
        train_step = optimizer.apply_gradients(zip(grads, weights))

        if config.get('optimizer_state_path'):
            pass # TODO(universome): load optimizer state

        initialize_uninitialized_variables(sess)

        batch_size = hp.get('batch_size', 16)
        epoch = 0
        training_start_time = time()
        loss_history = []
        val_scores = []

        def save_model():
            save_path = '{}/model.npz'.format(model_path)
            print('Saving the model into %s' %save_path)

            w_values = sess.run(weights)
            weights_dict = {w.name: w_val for w, w_val in zip(weights, w_values)}
            np.savez(save_path, **weights_dict)

        def save_optimizer_state(num_iters_done):
            # TODO(universome): Do we need iterations in optimizer state?
            state_dict = {var.name: sess.run(var) for var in non_trainable_vars}
            np.savez('{}/{}.iter-{}.npz'.format(model_path, 'optimizer_state', num_iters_done), **state_dict)

        num_iters_done = 0
        should_start_next_epoch = True # We need this var to break outer loop

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]



        for epoch in range(config.get('max_epochs', 1000000)):
            print("Beginning epoch", epoch)
            order = np.argsort([len(line) * (1 + np.random.normal(0, config.get('batch_length_variance', 0.2))) for line in src_train])

            batches = list(chunks(order, batch_size))
            shuffle(batches)
            with tqdm(batches) as t:
                for ids in t:
                    batch_src = [src_train[i] for i in ids]

                    # Note: we don't use voc.tokenize_many(batch, max_len=max_len)
                    # cuz it forces batch length to be that long and we often get away with much less
                    batch_src_ix = voc.tokenize_many(batch_src)[:, :max_len]

                    feed_dict = {inp: batch_src_ix}

                    loss_t = sess.run([train_step, loss], feed_dict)[1]
                    loss_history.append(np.mean(loss_t))

                    t.set_description('Iterations done: {}. Loss: {:.2f}'
                                      .format(num_iters_done, ewma(np.array(loss_history[-50:]), span=50)[-1]))

                    #if np.argmax(val_scores) == len(val_scores) - 1:


                    num_iters_done += 1

                    if config.get('max_time_seconds'):
                        seconds_elapsed = time()-training_start_time

                        if seconds_elapsed > config.get('max_time_seconds'):
                            print('Maximum allowed training time reached. Training took %s. Stopping.' % seconds_elapsed)
                            should_start_next_epoch = False
                            break
                print('Saving model')
                save_model()
                save_optimizer_state(num_iters_done)
                if not should_start_next_epoch:
                    break
                if config.get('max_epochs') and config.get('max_epochs') == epoch:
                    print('Maximum amount of epochs reached. Stopping.')
                    break

        save_model()
        save_optimizer_state(num_iters_done+1)


def main():
    parser = argparse.ArgumentParser(description='Run project commands')

    parser.add_argument('model')

    parser.add_argument('--data_path')
    parser.add_argument('--optimizer_state_path')
    parser.add_argument('--inp_embeddings_path')
    parser.add_argument('--out_embeddings_path')
    parser.add_argument('--target_lm_path')
    parser.add_argument('--src_lm_path')
    parser.add_argument('--pretrained_model_path')
    parser.add_argument('--hp_file_path')
    parser.add_argument('--use_early_stopping', type=bool)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_time_seconds', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--batch_length_variance', type=float)
    parser.add_argument('--gpu_memory_fraction', type=float)
    parser.add_argument('--lang', type=int)

    args = parser.parse_args()

    config = vars(args)
    config = dict(filter(lambda x: x[1], config.items())) # Getting rid of None vals

    print('Traning the %s' % args.model)
    print('Config:', config)
    train_model(args.model, config)


if __name__ == '__main__':
    main()
