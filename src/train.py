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
from batch_iterator import iterate_minibatches


def train_model(model_name, config):
    """Trains model, saves its weights and optimizer state"""

    model_path = 'trained_models/{}'.format(model_name)
    if not os.path.isdir('trained_models'): os.mkdir('trained_models')
    if not os.path.isdir(model_path): os.mkdir(model_path)

    src_train_path = '{}/bpe_parallel_train1.txt'.format(config.get('data_path'))
    dst_train_path = '{}/bpe_parallel_train2.txt'.format(config.get('data_path'))
    src_val_path = '{}/bpe_parallel_val1.txt'.format(config.get('data_path'))
    dst_val_path = '{}/bpe_parallel_val2.txt'.format(config.get('data_path'))

    src_train = open(src_train_path, 'r', encoding='utf-8').read().splitlines()
    dst_train = open(dst_train_path, 'r', encoding='utf-8').read().splitlines()
    src_val = open(src_val_path, 'r', encoding='utf-8').read().splitlines()
    dst_val = open(dst_val_path, 'r', encoding='utf-8').read().splitlines()

    inp_voc = Vocab.from_file('{}/1.voc'.format(config.get('data_path')))
    out_voc = Vocab.from_file('{}/2.voc'.format(config.get('data_path')))

    # Hyperparameters
    hp = json.load(open(config.get('hp_file'), 'r', encoding='utf-8')) if config.get('hp_file') else {}
    lr = hp.get('lr', 1e-4)

    use_early_stopping = hp.get('use_early_stopping', False)

    gpu_options = tf.GPUOptions(allow_growth=True)
    if config.get('gpu_memory_fraction'):
        gpu_options.per_process_gpu_memory_fraction=config.get('gpu_memory_fraction',0.95)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(model_name, inp_voc, out_voc, hp)

        inp = tf.placeholder(tf.int32, [None, None])
        out = tf.placeholder(tf.int32, [None, None])
        logprobs = model.symbolic_score(inp, out, is_train=True)[:,:tf.shape(out)[1]]

        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs, labels=out)
        loss = nll * infer_mask(out, out_voc.eos, dtype=tf.float32)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        non_trainable_vars = list(set(all_vars).difference(set(weights)))

        grads = tf.gradients(loss, weights)
        grads = tf.clip_by_global_norm(grads, 100)[0]
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.apply_gradients(zip(grads, weights))

        # Loading pretrained model
        if config.get('pretrained_model_path'):
            w_values = np.load(config.get('pretrained_model_path'))

            curr_var_names = set(w.name for w in weights)
            var_names_in_file = set(w_values.keys())

            assert curr_var_names == var_names_in_file

            assigns = [tf.assign(var, tf.constant(w_values[var.name])) for var in weights]
            sess.run(assigns)

        if config.get('optimizer_state_path'):
            pass # TODO(universome): load optimizer state

        # TODO(universome): embeddings will be in a different format
        if config.get('inp_embeddings_path'):
            embeddings = np.load(config.get('inp_embeddings_path'))['arr_0'].astype(np.float32)
            sess.run(tf.assign(model.emb_inp.trainable_weights[0], tf.constant(embeddings)))

        if config.get('out_embeddings_path'):
            embeddings = np.load(config.get('out_embeddings_path'))['arr_0'].astype(np.float32)
            sess.run(tf.assign(model.emb_out.trainable_weights[0], tf.constant(embeddings)))

        initialize_uninitialized_variables(sess)

        batch_size = config.get('batch_size', 16)
        epoch = 0
        loss_history = []
        val_scores = []

        def save_model(num_iters_done, is_last_model=False):
            if is_last_model:
                save_path = '{}/model.npz'.format(model_path)
            else:
                save_path = '{}/model.iter-{}.npz'.format(model_path, num_iters_done)

            print('Saving the model into %s' %save_path)

            w_values = sess.run(weights)
            weights_dict = {w.name: w_val for w, w_val in zip(weights, w_values)}
            np.savez(save_path, **weights_dict)

        # TODO(universome): this does not work, but looks like we do not need it :|
        if config.get('plot'):
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.show()

        num_iters_done = 0
        should_stop = False # We need this var to break outer loop

        while not should_stop:
            batches = batch_generator_over_dataset(src_train, dst_train, batch_size, batches_per_epoch=None)

            for batch_src, batch_dst in batches:
                batch_src_ix = inp_voc.tokenize_many(batch_src)
                batch_dst_ix = out_voc.tokenize_many(batch_dst)

                feed_dict = {inp: batch_src_ix, out: batch_dst_ix}

                loss_t = sess.run([train_step, loss], feed_dict)[1]
                loss_history.append(np.mean(loss_t))

                print('Iterations done: {}. Loss: {:.2f}'.format(num_iters_done, loss_t))

                if (num_iters_done+1) % config.get('save_every', 500) == 0:
                    save_model(num_iters_done+1)

                    # Saving optimizer state
                    state_dict = {var.name: sess.run(var) for var in non_trainable_vars}
                    np.savez('{}/{}.iter-{}.npz'.format(model_path, 'optimizer_state', num_iters_done+1), **state_dict)

                if (num_iters_done+1) % config.get('validate_every', 500) == 0:
                    print('Validating')
                    val_score = compute_bleu_for_model(model, sess, model.inp_voc, model.out_voc, src_val, dst_val, model_type=model_name)
                    val_scores.append(val_score)
                    print('Validation BLEU: {:0.3f}'.format(val_score))

                    if use_early_stopping and should_stop_early(val_scores, config.get('early_stopping_last_n')):
                        should_stop = True
                        print('Early stopping.')
                        break

                if config.get('plot') and (num_iters_done+1) % 10 == 0:
                    # figure(figsize=[8,8])
                    ax.clear()
                    ax.set_title('Batch loss')
                    ax.plot(loss_history)
                    ax.plot(ewma(np.array(loss_history), span=50))
                    ax.grid()
                    # fig.canvas.draw()
                    fig.show()

                num_iters_done += 1

                if config.get('max_num_iters') and num_iters_done == config.get('max_num_iters'):
                    should_stop = True
                    print('Maximum amount of iterations reached. Stopping.')
                    break

            epoch +=1

            if config.get('max_epochs') and config.get('max_epochs') == epoch:
                print('Maximum amount of epochs reached. Stopping.')
                break

        save_model(num_iters_done+1, True)


def main():
    parser = argparse.ArgumentParser(description='Run project commands')

    parser.add_argument('model')
    parser.add_argument('--data_path')
    parser.add_argument('--hp_file')
    parser.add_argument('--gpu_memory_fraction', type=float)
    parser.add_argument('--pretrained_model_path')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer_state_path')
    parser.add_argument('--validate_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--inp_embeddings_path')
    parser.add_argument('--out_embeddings_path')
    parser.add_argument('--max_num_iters', type=int)
    parser.add_argument('--use_early_stopping', type=bool)
    parser.add_argument('--early_stopping_last_n', type=int)
    parser.add_argument('--max_epochs', type=int)

    args = parser.parse_args()

    config = vars(args)
    config = dict(filter(lambda x: x[1], config.items())) # Getting rid of None vals

    print('Traning the %s' % args.model)
    train_model(args.model, config)


if __name__ == '__main__':
    main()
