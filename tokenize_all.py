import sys
import argparse
import nltk

import numpy as np
from sklearn.model_selection import train_test_split

nltk.download('punkt')

parser = argparse.ArgumentParser()

parser.add_argument(
        '-o', type=str, dest='out_path')
parser.add_argument(
        '-i', type=str, dest='inp_path')
parser.add_argument(
        '-r', type=float, dest='fraction', default = 0.2)

args = parser.parse_args()


path = "/nmt/nltk"
nltk.download('punkt', path)
nltk.data.path.append(path)


with open(args.inp_path + '/parallel_corpus.txt', encoding = 'utf-8') as f:
    lines = [x[:-1] for x in f.readlines()]


lines_train, lines_val = train_test_split(lines, test_size=args.fraction, shuffle=True, random_state=42)

def tok_line(line):
    return ' '.join(nltk.word_tokenize(line))


def tok_split_several_lines(corpus, n):
    return '\n'.join([tok_line(x.split('\t')[n]) for x in corpus]).encode('utf-8')


def tok_several_lines(corpus):
    return '\n'.join([tok_line(x) for x in corpus]).encode('utf-8')

with open(args.out_path + '/tok_parallel_train1.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines_train, 0))
    # f.write(tok(lines_train, 0))
with open(args.out_path + '/tok_parallel_train2.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines_train, 1))

with open(args.out_path + '/tok_parallel_val1.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines_val, 0))
with open(args.out_path + '/tok_parallel_val2.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines_val, 1))

with open(args.out_path + '/tok_parallel1.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines, 0))
with open(args.out_path + '/tok_parallel2.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines, 1))

with open(args.inp_path + '/parallel_corpus.txt', encoding = 'utf-8') as f:
    lines = [x[:-1] for x in f.readlines()]


###
with open(args.inp_path + '/corpus1.txt', encoding = 'utf-8') as f:
    lines1 = [x[:-1] for x in f.readlines()]
with open(args.inp_path + '/corpus2.txt', encoding = 'utf-8') as f:
    lines2 = [x[:-1] for x in f.readlines()]
with open(args.inp_path + '/input.txt', encoding='utf-8') as f:
    lines_inp = [x[:-1] for x in f.readlines()]

with open(args.out_path + '/tok_corpus1.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines1))
with open(args.out_path + '/tok_corpus2.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines2))
with open(args.out_path + '/tok_input.txt', 'wb') as f:
    f.write(tok_split_several_lines(lines_inp))