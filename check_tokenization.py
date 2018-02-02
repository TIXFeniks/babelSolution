import sys

inputs_bpe_path = sys.argv[1]
inputs_path = sys.argv[2]

inputs_bpe_lines = open(inputs_bpe_path, 'r', encoding='utf-8').read().splitlines()
inputs_lines = open(inputs_path, 'r', encoding='utf-8').read().splitlines()


if len(inputs_bpe_lines) != len(inputs_lines):
    print("line number mismatch!")
    exit(1)

for line_bpe, line in zip(inputs_bpe_lines, inputs_lines):
    if len(line_bpe.split()) < len(line.split()):
        print("line length adequacy check failed!")
        exit(1)

