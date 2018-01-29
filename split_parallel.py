#!/usr/bin/env python

import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
        '-o', type=str, dest='path')

args = parser.parse_args()

lines = [] 
for line in sys.stdin:
    lines.append(line) 

print(args.path + "/parallel1.txt")
with open(args.path + "/parallel1.txt", 'w') as f:
    f.write('\n'.join([x.split('\t')[0] for x in lines]))
with open(args.path + "/parallel2.txt", 'w') as f:
    f.write('\n'.join([x.split('\t')[1] for x in lines]))



