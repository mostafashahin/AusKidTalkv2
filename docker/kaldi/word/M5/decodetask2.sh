sox -t wav - /opt/tmp/tmpSpeech.wav
. ./path.sh
model=model
ivector_ext=model/iv_extractor/
lm_graph=model/graph_task2_tedlium_tgsmall_0.25/ 
lm_model= #Needed in case of lm rescore
rnn_rescore_lm=
lm_rescore= 
mfcc_config=model/conf/mfcc_hires.conf

speech_file=/opt/tmp/tmpSpeech.wav

rand=$RANDOM
clean=true
win=0.0
lmscal=12.0
align_ph=false
wrk_dir="wrk_child-$rand"
mkdir $wrk_dir
#prepare data dir from speech file
mkdir -p $wrk_dir/data
basename=`basename $speech_file`
echo "$basename sox $speech_file -t wav -b 16 -r 16000 -c 1 - |" > $wrk_dir/data/wav.scp
echo spk1 $basename > $wrk_dir/data/spk2utt
echo $basename spk1 > $wrk_dir/data/utt2spk

utils/copy_data_dir.sh $wrk_dir/data $wrk_dir/data_hires > /dev/null 2>&1
steps/make_mfcc.sh --nj 1 --mfcc-config $mfcc_config $wrk_dir/data_hires > /dev/null 2>&1
steps/compute_cmvn_stats.sh $wrk_dir/data_hires > /dev/null 2>&1
utils/fix_data_dir.sh $wrk_dir/data_hires > /dev/null 2>&1

nspk=$(wc -l <$wrk_dir/data_hires/spk2utt)
steps/online/nnet2/extract_ivectors_online.sh --nj "${nspk}" $wrk_dir/data_hires $ivector_ext $wrk_dir/ivectors_data_hires > /dev/null 2>&1

steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 1 --extra-left-context 0 --extra-right-context 0 --extra-left-context-initial 0 --extra-right-context-final 0  --frames-per-chunk 140  --skip_diagnostics true --skip_scoring true --online-ivector-dir $wrk_dir/ivectors_data_hires $lm_graph $wrk_dir/data_hires $model/decode_data_$rand > /dev/null 2>&1

lattice-add-penalty --word-ins-penalty=$win "ark:gunzip -c $model/decode_data_$rand/lat.1.gz|" ark:- 2>/dev/null | lattice-push ark:- ark:- 2>/dev/null | lattice-align-words --silence-label=105090 $lm_graph/phones/word_boundary.int $model/final.mdl ark:- ark:$model/decode_data_$rand/ali.wrd.lat 2>/dev/null



if [ ! -z $rnn_rescore_lm ]; then

	rnnlm/lmrescore_pruned.sh --weight 0.45 --max-ngram-order 4 $lm_model $rnn_rescore_lm $wrk_dir/data_hires $model/decode_data_$rand $model/decode_data_${rand}_rnn_rescore >/dev/null 2>&1
	lattice-add-penalty --word-ins-penalty=$win "ark:gunzip -c $model/decode_data_${rand}_rnn_rescore/lat.1.gz|" ark:- 2>/dev/null | lattice-push ark:- ark:- 2>/dev/null | lattice-align-words --silence-label=105090 $lm_graph/phones/word_boundary.int $model/final.mdl ark:- ark:$model/decode_data_$rand/ali.wrd.lat 2>/dev/null
fi

if [ ! -z $lm_rescore ]; then
	steps/lmrescore_const_arpa.sh $lm_model $lm_rescore $wrk_dir/data_hires $model/decode_data_$rand $model/decode_data_${rand}_lm_rescore >/dev/null 2>&1
	lattice-add-penalty --word-ins-penalty=$win "ark:gunzip -c $model/decode_data_${rand}_lm_rescore/lat.1.gz|" ark:- 2>/dev/null | lattice-push ark:- ark:- 2>/dev/null | lattice-align-words --silence-label=105090 $lm_graph/phones/word_boundary.int $model/final.mdl ark:- ark:$model/decode_data_$rand/ali.wrd.lat 2>/dev/null
fi

lattice-to-ctm-conf --decode-mbr=true --frame-shift=0.03 --inv-acoustic-scale=$lmscal ark:$model/decode_data_$rand/ali.wrd.lat - 2>/dev/null | int2sym.pl -f 5 $lm_graph/words.txt > $basename-child-tedlium-$rand 2>/dev/null

if $align_ph; then
	lattice-to-phone-lattice $model/final.mdl ark:$model/decode_data_$rand/ali.wrd.lat ark:- 2>/dev/null | lattice-to-ctm-conf --decode-mbr=true --frame-shift=0.03 --inv-acoustic-scale=10.0 ark:- - 2>/dev/null | int2sym.pl -f 5 $lm_graph/phones.txt > phone.ali 2>/dev/null
fi
if $clean; then
	rm -r $wrk_dir
	rm -r $model/decode_data_$rand
fi

cat $basename-child-tedlium-${rand}
rm $basename-child-tedlium-${rand}
