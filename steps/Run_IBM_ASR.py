import sys, os
sys.path.insert(0,os.path.abspath('tools/'))

import stt.ibm.stt as ibm_stt
import txtgrid_master.TextGrid_Master as tgm
import librosa
import soundfile as sf
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from os.path import join, isfile

#TODO replace with argparse
dir = sys.argv[1]
childID = sys.argv[2]
taskID = sys.argv[3]
modelNameString = sys.argv[4]

if not os.path.isdir(dir):
    #print("{} is not exist".format(dir))
    sys.exit("{} is not exist".format(dir))
    
tmp_dir = os.path.join(dir,'tmp')
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)

#TODO check wav file is exist

wav_file = os.path.join(dir,'{}_{}.wav'.format(childID, taskID))
json_file = os.path.join(dir,'{}_{}_ibm.json'.format(childID, taskID))
txtgrid_file = os.path.join(dir,'{}_{}_ibm.TextGrid'.format(childID, taskID))

modelNames = modelNameString.split()
masterModelName = modelNames[0]

#Get duration
dur_in_secs = librosa.get_duration(filename=wav_file)
bThreeSpeaker = False
if isfile(json_file):
    with open(json_file) as fjson:
        results = json.load(fjson)
else:
    lResults={}
    print('child: {} task: {} diarise modelString {}'.format(childID, taskID, modelNameString))
    for modelName in modelNames:
        print('child: {} task: {} diarise using model {}'.format(childID, taskID, modelName))
        if not bThreeSpeaker:
            results = ibm_stt.stt_audio_file_wav(wav_file,model_str=modelName)
            n_spkrs = pd.DataFrame.from_records(results[0]['speaker_labels']).speaker.unique().shape[0]
            json_file_model = os.path.join(dir,'{}_{}_{}_ibm.json'.format(childID, taskID, modelName))
            with open(json_file_model,'w') as fjson:
                json.dump(results, fjson)
            lResults[modelName] = results
            print('child: {} task: {} model {} got {} speakers'.format(childID, taskID, modelName, n_spkrs))
            if n_spkrs == 3:
                bThreeSpeaker = True
                with open(json_file,'w') as fjson:
                    json.dump(results, fjson)
    if not bThreeSpeaker:
        results = lResults[masterModelName]
        with open(json_file,'w') as fjson:
            json.dump(results, fjson)

df_spkrs = pd.DataFrame.from_records(results[0]['speaker_labels'])

timestamps=[]
_ = [timestamps.extend(x['alternatives'][0]['timestamps']) for x in results[0]['results']]
df_words = pd.DataFrame.from_records(timestamps,columns=['word','from','to'])

df_results = pd.merge(df_spkrs, df_words, on=['from','to'])

n_spkrs = df_results.speaker.unique().shape[0]
print('child: {} task: {} has {} speakers detected'.format(childID, taskID, n_spkrs))
tiersDict = {}
for spk in df_results.speaker.unique():
    tierName = 'spk{}'.format(spk)
    df_results_spk = df_results[df_results.speaker==spk]
    stTime, endTime, labels = [a.squeeze() for a in np.split(df_results_spk[['from','to','word']].values,3,1)]
    tiersDict[tierName] = (list(stTime), list(endTime), list(labels))

tgm.WriteTxtGrdFromDict(txtgrid_file,tiersDict,0,dur_in_secs,sFilGab='')


