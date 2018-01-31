import os
import json
import argparse
# import subprocess

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import ewma
from sklearn.model_selection import train_test_split

from bleu import compute_bleu
from models.gnmt_lstm import AttentiveTranslationModel
from vocab import Vocab
from src.training_utils import batch_generator_over_dataset, compute_bleu_for_model
from lib.tensor_utils import infer_mask, initialize_uninitialized_variables
from batch_iterator import iterate_minibatches

MODEL_NAME = 'super_gnmt_model'

def train_gnmt(config):
    """
    Trains GNMT model, saves its weights and optimizer state
    """

    model_path = 'trained_models/{}'.format(MODEL_NAME)
    if not os.path.isdir('trained_models'): os.mkdir('trained_models')
    if not os.path.isdir(model_path): os.mkdir(model_path)

    src_path = '{}/bpe_parallel1.txt'.format(config.get('data_path'))
    dst_path = '{}/bpe_parallel2.txt'.format(config.get('data_path'))

    src_data = open(src_path, 'r', encoding='utf-8').read().splitlines()
    dst_data = open(dst_path, 'r', encoding='utf-8').read().splitlines()

    split = train_test_split(src_data, dst_data, test_size=config.get('val_split_size', 0.1), random_state=42)
    src_train, src_val, dst_train, dst_val = split

    inp_voc = Vocab.from_file('{}/1.voc'.format(config.get('data_path')))
    out_voc = Vocab.from_file('{}/2.voc'.format(config.get('data_path')))

    # Hyperparameters
    hp = json.load(open(config.get('hp_file'), 'r', encoding='utf-8')) if config.get('hp_file') else {}
    emb_size = hp.get('emb_size', 32)
    hid_size = hp.get('hid_size', 32)
    attn_size = hp.get('attn_size', 32)
    lr = hp.get('lr', 1e-4)
    use_early_stopping = hp.get('use_early_stopping', False)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.get('gpu_memory_fraction', 0.3))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = AttentiveTranslationModel(MODEL_NAME, inp_voc, out_voc, emb_size, hid_size, attn_size)

        inp = tf.placeholder(tf.int32, [None, None])
        out = tf.placeholder(tf.int32, [None, None])
        logprobs = model.symbolic_score(inp, out, is_train=True)[:,:tf.shape(out)[1]]

        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs, labels=out)
        loss = nll * infer_mask(out, out_voc.eos, dtype=tf.float32)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, MODEL_NAME)

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        non_trainable_vars = list(set(all_vars).difference(set(weights)))

        grads = tf.gradients(loss, weights)
        grads = tf.clip_by_global_norm(grads, 100)[0]
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.apply_gradients(zip(grads, weights))

        # Loading pretrained model
        if config.get('pretrained_model_path'):
            # model_weights_path = open(config.get('pretrained_model_path'), 'rb')
            w_values = np.load(config.get('pretrained_model_path'))

            curr_var_names = set(w.name for w in weights)
            var_names_in_file = set(w_values.keys())

            assert curr_var_names == var_names_in_file

            assigns = [tf.assign(var, tf.constant(w_values[var.name])) for var in weights]
            sess.run(assigns)

        if config.get('optimizer_state_path'):
            pass # TODO(universome): load optimizer state

        if config.get('inp_embeddings_path'):
            embeddings = np.load(config.get('inp_embeddings_path'))['arr_0'].astype(np.float32)
            sess.run(tf.assign(model.emb_inp.trainable_weights[0], tf.constant(embeddings)))

        if config.get('out_embeddings_path'):
            embeddings = np.load(config.get('out_embeddings_path'))['arr_0'].astype(np.float32)
            sess.run(tf.assign(model.emb_out.trainable_weights[0], tf.constant(embeddings)))

        initialize_uninitialized_variables(sess)

        batch_size = config.get('batch_size', 16)
        batches = batch_generator_over_dataset(src_train, dst_train, batch_size, batches_per_epoch=None)
        loss_history = []
        val_scores = []

        # TODO(universome): this does not work, but looks like we do not need it :|
        if config.get('plot'):
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.show()

        while True:
            for i, (batch_src, batch_dst) in enumerate(tqdm(batches)):
                batch_src_ix = inp_voc.tokenize_many(batch_src)
                batch_dst_ix = out_voc.tokenize_many(batch_dst)

                feed_dict = {inp: batch_src_ix, out: batch_dst_ix}

                loss_t = sess.run([train_step, loss], feed_dict)[1]
                loss_history.append(np.mean(loss_t))

                print('Iterations done: {}. Loss: {:.2f}'.format(i, loss_t))

                if (i+1) % config.get('save_every', 500) == 0:
                    # Saving model
                    w_values = sess.run(weights)
                    weights_dict = {w.name: w_val for w, w_val in zip(weights, w_values)}
                    np.savez('{}/model.iter-{}.npz'.format(model_path, i+1), **weights_dict)

                    # Saving optimizer state
                    state_dict = {var.name: sess.run(var) for var in non_trainable_vars}
                    np.savez('{}/{}.iter-{}.npz'.format(model_path, 'optimizer_state', i+1), **state_dict)

                if (i+1) % config.get('validate_every', 500) == 0:
                    print('Validating')
                    val_score = compute_bleu_for_model(model, sess, model.inp_voc, model.out_voc, src_val, dst_val)
                    val_scores.append(val_score)
                    print('Validation BLEU: {:0.3f}'.format(val_score))

                    if use_early_stopping and len(val_scores) > 0 and val_scores[-1] < val_score[-2]:
                        break

                if config.get('plot') and (i+1) % 10 == 0:
                    # figure(figsize=[8,8])
                    ax.clear()
                    ax.set_title('Batch loss')
                    ax.plot(loss_history)
                    ax.plot(ewma(np.array(loss_history), span=50))
                    ax.grid()
                    # fig.canvas.draw()
                    fig.show()

            epoch +=1

            if config.get('max_epochs') and config.get('max_epochs') == epoch:
                break

def main():
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='model')

    model_parser = subparsers.add_parser('gnmt')

    model_parser.add_argument('--data_path')
    model_parser.add_argument('--hp_file')
    model_parser.add_argument('--gpu_memory_fraction', type=float)
    model_parser.add_argument('--pretrained_model_path')
    model_parser.add_argument('--batch_size', type=int)
    model_parser.add_argument('--optimizer_state_path')
    model_parser.add_argument('--validate_every', type=int)
    model_parser.add_argument('--save_every', type=int)
    model_parser.add_argument('--val_split_size', type=float)
    model_parser.add_argument('--inp_embeddings_path')
    model_parser.add_argument('--out_embeddings_path')
    model_parser.add_argument('--max_epochs', type=int)

    args = parser.parse_args()

    if args.model == 'gnmt':
        print('Training gnmt!')

        config = vars(args)
        config = dict(filter(lambda x: x[1], config.items())) # Getting rid of None vals

        train_gnmt(config)
    else:
        print('No model name is provided')

if __name__ == '__main__':
    main()
