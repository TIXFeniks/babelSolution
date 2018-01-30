import sys
import argparse

import fasttext

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

print(args)
model_1_bpe = fasttext.cbow(args.inp_path + "/bpe_all_1.txt", args.out_path + "/model_1_bpe", dim=emb_dim, thread=4, epoch=args.epochs, bucket = 10)
model_2_bpe = fasttext.cbow(args.inp_path + "/bpe_all_2.txt", args.out_path + "/model_2_bpe", dim=emb_dim, thread=4, epoch=args.epochs, bucket = 10)




