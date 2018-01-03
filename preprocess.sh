#!/bin/sh
# usage: $1 - src, $2 - output dir

#suffix of source language
SRC=$3

# suffix of target language
TRG=$4

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=lib/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=lib/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=lib/nematus

# theano device
device=cpu

# preprocess
cat $1 | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
$mosesdecoder/scripts/recaser/truecase.perl -model trans/truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c trans/$SRC$TRG.bpe > $2/"${1##*/}".bpe 


#lib/subword-nmt/learn_bpe.py < $1 > $2/"${1##*/}".voc
#lib/subword-nmt/apply_bpe.py -c $2/"${1##*/}".voc < $1 > $2/"${1##*/}".bpe
