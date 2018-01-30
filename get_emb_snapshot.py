import sys
import argparse
import fasttext
import numpy as np

from vocab import Vocab


parser = argparse.ArgumentParser()

parser.add_argument(
        '-i', type=str, dest='inp_path', required=True)
parser.add_argument(
        '-o', type=str, dest='out_path', required=True)
parser.add_argument(
        '--inp_emb_name', type=str, dest='inp_emb_name', default='transformer_inp_emb')
parser.add_argument(
        '--out_emb_name', type=str, dest='out_emb_name', default='transformer_out_emb')
parser.add_argument(
        '-d', type=int, dest='emb_dim', default=300)
parser.add_argument(
        '-e', type=int, dest='epochs', default=3)

args = parser.parse_args()

emb_dim = args.emb_dim
epochs = 1

print("Starting training models...")
model_1_bpe = fasttext.cbow(args.inp_path + "/bpe_all_1.txt", args.out_path + "/model_1_bpe", dim=emb_dim, thread=4, epoch=args.epochs)
model_2_bpe = fasttext.cbow(args.inp_path + "/bpe_all_2.txt", args.out_path + "/model_2_bpe", dim=emb_dim, thread=4, epoch=args.epochs)

print("Making embeddings")
voc_1 = Vocab.from_file(args.inp_path + "/1.voc")
voc_2 = Vocab.from_file(args.inp_path + "/2.voc")

emb_1 = np.vstack([np.array(model_1_bpe[w]) for w in voc_1.tokens])
emb_2 = np.vstack([np.array(model_2_bpe[w]) for w in voc_2.tokens])

np.savez(args.out_path + "emb_snapshot", **{args.inp_emb_name : emb_1, args.out_emb_name : emb_2})


