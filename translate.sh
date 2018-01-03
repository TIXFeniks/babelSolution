#!/bin/sh

# this sample script translates a test set, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# and postprocessing (merging subword units, detruecasing, detokenization).

# instructions: set paths to mosesdecoder, subword_nmt, and nematus,
# then run "./translate.sh < input_file > output_file"

# suffix of source language
SRC=en

# suffix of target language
TRG=de

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=lib/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=lib/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=lib/nematus

# theano device
device=cpu

# preprocess
cat /data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe
# translate
#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,base_compiledir=. python $nematus/nematus/translate.py \
#     -m model.npz \
#     -k 12 -n -p 10 --suppress-unk | \
# postprocess
#sed 's/\@\@ //g' | \
#$mosesdecoder/scripts/recaser/detruecase.perl | \
#$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG >/output/output.txt
