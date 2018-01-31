#!/bin/sh

### USAGE: ./tokenize.sh [home_folder (str)] [tokens (int)] [joint dict (true/false)]
### home_folder should contain ./data ./libs/mosesdecoder ./libs/subword-nmt ./scripts/split_parallel.py

if [ -z "$1" ]; then
	home=/home/anton/deephack/onsight
else
	home=$1
fi

if [ -z "$2" ]; then
	data_dir=/home/anton/deephack/onsight/data
else
	data_dir=$2
fi

if [ -z "$3" ]; then
	tokens=4000
else
	tokens=$3
fi

if [ -z "$4" ]; then
	joint_dict=false
else
	joint_dict=$4
fi


mosesdecoder=$home/ext_libs/mosesdecoder
subword_nmt=$home/ext_libs/subword-nmt
inp_data=$data_dir
data=$home/data

mkdir $data

threads=4

echo "$joint_dict"
echo "$home"
echo "$tokens"

python3.6 $home/split_parallel.py -o $data -i $inp_data

cp $inp_data/corpus1.txt $data/
cp $inp_data/corpus2.txt $data/
cp $inp_data/input.txt $data/


for lang in 1 2
do
	for corp in parallel_train parallel_val corpus
	do
		echo "Tokenizing ${corp}${lang} ..."
		cat $data/$corp$lang.txt | \
		$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
		$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads $threads -penn > \
		$data/tok_$corp$lang.txt
	done
	(cat $data/tok_parallel_val$lang.txt; cat $data/tok_parallel_train$lang.txt; cat $data/tok_corpus$lang.txt) > $data/tok_all_$lang.txt
done



if [ $joint_dict = true ]; then
	echo "Learning joint ..."
	$subword_nmt/learn_joint_bpe_and_vocab.py -i $data/tok_all_1.txt $data/tok_all_2.txt -s $tokens -o $data/1.bpe --write-vocabulary $data/1.voc $data/2.voc
	cp $data/1.bpe $data/2.bpe
else
	for lang in 1 2
	do
		echo "Learning voc ${lang} ..."
		$subword_nmt/learn_joint_bpe_and_vocab.py -i $data/tok_all_$lang.txt -s $tokens -o $data/$lang.bpe --write-vocabulary $data/$lang.voc
	done
fi



echo 'Transforming'
for lang in 1 2
do
	for corp in parallel_train parallel_val corpus
	do
		echo "Transforming ${corp} ${lang} ..."
		cat $data/tok_$corp$lang.txt | \
		$subword_nmt/apply_bpe.py -c $data/$lang.bpe --vocabulary $data/$lang.voc \
		--vocabulary-threshold 0 -o $data/bpe_$corp$lang.txt
	done
	(cat $data/bpe_parallel_train$lang.txt; cat $data/bpe_parallel_val$lang.txt; cat $data/bpe_corpus$lang.txt) > $data/bpe_all_$lang.txt
done

cat $data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -threads=$threads -penn > \
$data/tok_input.txt

cat $data/tok_input.txt | \
$subword_nmt/apply_bpe.py -c $data/$lang.bpe --vocabulary $data/$lang.voc \
--vocabulary-threshold 0 -o $data/bpe_input.txt
