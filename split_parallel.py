import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
        '-o', type=str, dest='out_path')
parser.add_argument(
        '-i', type=str, dest='inp_path')
parser.add_argument(
        '-r', type=float, dest='fraction', default = 0.2)

args = parser.parse_args()

with open(args.inp_path + '/parallel_corpus.txt', encoding = 'utf-8') as f:
    lines = [x[:-1] for x in f.readlines()]

np.random.seed(322)
np.random.shuffle(lines)
lines_num = len(lines)

lines_train = lines[:round(lines_num * (1 - args.fraction))]
lines_val = lines[round(lines_num * (1 - args.fraction)):]


with open(args.out_path + '/parallel_train1.txt', 'wb') as f:
    f.write('\n'.join([x.split('\t')[0] for x in lines_train]).encode('utf-8'))
with open(args.out_path + '/parallel_train2.txt', 'wb') as f:
    f.write(''.join([x.split('\t')[1] for x in lines_train]).encode('utf-8'))
with open(args.out_path + '/parallel_val1.txt', 'wb') as f:
    f.write('\n'.join([x.split('\t')[0] for x in lines_val]).encode('utf-8'))
with open(args.out_path + '/parallel_val2.txt', 'wb') as f:
    f.write(''.join([x.split('\t')[1] for x in lines_val]).encode('utf-8'))
