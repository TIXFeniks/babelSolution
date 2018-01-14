#!/bin/sh


./preprocess.sh $1 /tmp en de


THEANO_FLAGS=device=cpu python translate.py /tmp/"${1##*/}".bpe 
