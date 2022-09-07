wget -O scripts/words.map https://raw.githubusercontent.com/mostafashahin/AusKidTalkv2/main/scripts/words.map
#TODO Add the models to the config file
ibmModelString="Multimedia Telephony"
cat - | while read id
do
	if [ ! -f /opt/data/$id/txtgrids/${id}_task1_ibm.TextGrid ]; then
	    python3 steps/Run_IBM_ASR.py /opt/data/$id/txtgrids/ $id task1 "$ibmModelString" 2>&1 | tee -a /opt/data/$id/asr.log
	fi
	if [ ! -f /opt/data/$id/txtgrids/${id}_task1_kaldi.TextGrid ]; then
	    python3 steps/Run_Kaldi_on_IBM.py /opt/data/$id/txtgrids/ $id task1 scripts/words.map 2>&1 | tee -a /opt/data/$id/asr.log
	fi
done
