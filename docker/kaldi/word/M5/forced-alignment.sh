sox -t wav - /opt/tmp/tmpSpeech.wav

. ./path.sh
model=model
ivector_ext=model/iv_extractor/
lm_graph=model/graph_tedlium_tgsmall/
lang=model/lang/
lm_model= #Needed in case of lm rescore
rnn_rescore_lm=
lm_rescore= 
mfcc_config=model/conf/mfcc_hires.conf

speech_file=/opt/tmp/tmpSpeech.wav

text=

rand=$RANDOM
clean=true
win=0.0
lmscal=12.0
align_ph=false
wrk_dir="wrk_child-$rand"

. ./utils/parse_options.sh
mkdir $wrk_dir
#prepare data dir from speech file
mkdir -p $wrk_dir/data
basename=`basename $speech_file`
echo "$basename sox $speech_file -t wav -b 16 -r 16000 -c 1 - |" > $wrk_dir/data/wav.scp
echo "$basename $text" > $wrk_dir/data/text
echo spk1 $basename > $wrk_dir/data/spk2utt
echo $basename spk1 > $wrk_dir/data/utt2spk

utils/copy_data_dir.sh $wrk_dir/data $wrk_dir/data_hires > /dev/null 2>&1
steps/make_mfcc.sh --nj 1 --mfcc-config $mfcc_config $wrk_dir/data_hires > /dev/null 2>&1
steps/compute_cmvn_stats.sh $wrk_dir/data_hires > /dev/null 2>&1
utils/fix_data_dir.sh $wrk_dir/data_hires > /dev/null 2>&1

nspk=$(wc -l <$wrk_dir/data_hires/spk2utt)
steps/online/nnet2/extract_ivectors_online.sh --nj "${nspk}" $wrk_dir/data_hires $ivector_ext $wrk_dir/ivectors_data_hires > /dev/null 2>&1

steps/nnet3/align.sh --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' --online-ivector-dir $wrk_dir/ivectors_data_hires --use-gpu false --nj 1 $wrk_dir/data_hires $lang $model $wrk_dir/align > /dev/null 2>&1

ali-to-phones --frame-shift=0.03 --ctm-output $model/final.mdl ark:"gunzip -c $wrk_dir/align/ali.1.gz |" - 2>/dev/null | utils/int2sym.pl -f 5 $model/phones_noP.txt - 2>/dev/null > $wrk_dir/align/ali.1.ctm 

cat $wrk_dir/align/ali.1.ctm