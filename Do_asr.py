import google.stt as g_stt
import aws.stt as aws_stt
import ibm.stt as ibm_stt
import os
from os.path import join,splitext,isfile
from glob import glob
import pandas as pd
import json


#wavFilesPath = '/media/Windows/root/dp_project/dp_asr_errors_stimulus_sil'
wavFilesPath = '/media/Windows/root/dp_project/Aus_with_carrier'
lang = 'en-AU'
asr = 'google'
res_file = '/media/Windows/root/dp_project/asr_results_carrier_all.csv'
df = pd.read_csv(res_file, header=0)

wavFiles = glob(join(wavFilesPath,'*.wav'))
if asr == 'google':
    config = g_stt.config
    config['language_code'] = lang
    for wavFile in wavFiles:
        
        response_file = splitext(wavFile)[0]+'.'+lang+'.google.json'
        if isfile(response_file):
            print('response file found ...')
            with open(response_file,'r') as jf:
                phrase = json.load(jf)
            g_out = g_stt.speech.RecognizeResponse.from_json(phrase)
        else:
            g_out = g_stt.stt_single_file(wavFile,config,overwrite=False)

        if g_out:
            transcript = g_out.results[0].alternatives[0].transcript
            conf_score = g_out.results[0].alternatives[0].confidence
            with open(response_file, 'w') as jf:
                json.dump(type(g_out).to_json(g_out), jf)
        else:
            transcript=''
            conf_score=''

        item  = {'File_Name':wavFile, 'ASR':asr, 'ASR_lang':lang, 'Transcription':transcript, 'Confidence_score':conf_score}
        df = df.append(item, ignore_index=True)
df.to_csv(res_file,index=False)
