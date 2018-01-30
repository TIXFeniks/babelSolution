#!/bin/sh

if [ -z "$1" ]; then
	home=/home/anton/deephack/onsight  
else
	home=$1
fi

mosesdecoder=$home/libs/mosesdecoder
subword_nmt=$home/libs/subword-nmt
scripts=$home/scripts
data=$home/data

tokens=4000
threads=8

cat $data/parallel_corpus.txt | $scripts/split_parallel.py -o $data

for lang in 1 2
do
	for corp in parallel corpus
	do 	
		echo "Tokenizing ${corp}${lang} ..."
		cat $data/$corp$lang.txt | \
		$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
		$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads $threads -penn > \
		$data/tok_$corp$lang.txt
	done
	(cat $data/tok_parallel$lang.txt; cat $data/tok_corpus$lang.txt) > $data/tok_all_$lang.txt
done

for lang in 1 2
do 
	echo "Learning voc ${lang} ..."
	$subword_nmt/learn_joint_bpe_and_vocab.py -i $data/tok_all_$lang.txt -s $tokens -o $data/$lang.bpe --write-vocabulary $data/$lang.voc
done

echo 'Transforming'
for lang in 1 2
do
	for corp in parallel corpus
	do 
		echo "Transforming ${corp} ${lang} ..."
		cat $data/tok_$corp$lang.txt | \
		$subword_nmt/apply_bpe.py -c $data/$lang.bpe --vocabulary $data/$lang.voc \
		--vocabulary-threshold 0 -o $data/bpe_$corp$lang.txt
	done
	(cat $data/bpe_parallel$lang.txt; cat $data/bpe_corpus$lang.txt) > $data/bpe_all_$lang.txt
done

cat $data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads=$threads -penn > \
$data/tok_input.txt

cat $data/tok_input.txt | \
$subword_nmt/apply_bpe.py -c $data/$lang.bpe --vocabulary $data/$lang.voc \
--vocabulary-threshold 0 -o $data/bpe_input.txt
