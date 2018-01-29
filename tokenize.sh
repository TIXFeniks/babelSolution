#!/bin/sh

home=/home/anton/deephack/onsight
mosesdecoder=$home/libs/mosesdecoder
subword_nmt=$home/libs/subword-nmt
scripts=$home/scripts
data=$home/data

iters=32000

cat $data/parallel_corpus.txt | $scripts/split_parallel.py -o $data

for lang in 1 2
do 
	echo 'learning voc $lang'
	cat $data/parallel$lang.txt $data/corpus$lang.txt | \
	$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
	$mosesdecoder/scripts/tokenizer/tokenizer.perl -penn > \
	$data/all_tok_$lang.txt


	$subword_nmt/learn_joint_bpe_and_vocab.py -i $data/all_tok_$lang.txt -s 32000 -o $data/$lang.bpe --write-vocabulary $data/$lang.voc
done


echo 'transforming'
for lang in 1 2
do
    for corp in parallel, corpus
	do 
		echo 'transforming $corp $lang'
		cat $data/$corp$lang.txt | \
		$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
		$mosesdecoder/scripts/tokenizer/tokenizer.perl -penn | \
		$subword_nmt/apply_bpe.py -c $data/$lang.bpe --vocabulary $data/$lang.voc \
		--vocabulary-threshold 0 -o $data/bpe_$corp$lang.txt
	done
done

#TRANSFORM CORPUSES
#cat $data/corpus1_short.txt | \
#$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
#$mosesdecoder/scripts/tokenizer/tokenizer.perl -penn | \
#$subword_nmt/apply_bpe.py -c $data/1.bpe --vocabulary $data/1.voc \
#--vocabulary-threshold 0 -o $data/corpus1_bpe.txt



