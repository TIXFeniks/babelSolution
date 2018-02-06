import os
import sys


tok_path = sys.argv[1]
out_path = sys.argv[2]

tok_lines = open(tok_path, 'r', encoding='utf-8').read().splitlines()
out_lines = open(out_path, 'r', encoding='utf-8').read().splitlines()


if len(tok_lines) - len(out_lines) > -2:
    print("ACHTUNG! Detokenization failed")
    os.system("mv %s %s" % (tok_path, out_path))
