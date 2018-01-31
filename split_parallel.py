import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
        '-o', type=str, dest='path')
parser.add_argument(
        '-r', type=float, dest='fraction', default = 0.2)

args = parser.parse_args()

lines = [] 
for line in sys.stdin:
    lines.append(line) 

np.random.seed(322)
np.random.shuffle(lines)
lines_num = len(lines)

lines_train = lines[:round(lines_num * (1 - args.fraction))]
lines_val = lines[round(lines_num * (1 - args.fraction)):]


with open(args.path + "/parallel_train1.txt", 'w') as f:
    f.write('\n'.join([x.split('\t')[0] for x in lines_train]))
with open(args.path + "/parallel_train2.txt", 'w') as f:
    f.write(''.join([x.split('\t')[1] for x in lines_train]))
with open(args.path + "/parallel_val1.txt", 'w') as f:
    f.write('\n'.join([x.split('\t')[0] for x in lines_val]))
with open(args.path + "/parallel_val2.txt", 'w') as f:
    f.write(''.join([x.split('\t')[1] for x in lines_val]))
