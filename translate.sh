#!/bin/sh


./preprocess.sh input/input.txt input/ en de


THEANO_FLAGS=device=cpu python translate.py
