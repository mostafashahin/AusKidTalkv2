from os.path import basename, splitext, isfile, join, isdir
import sys, glob
sys.path.insert(0,'tools')
from os import makedirs
import configparser, argparse
import txtgrid_master.TextGrid_Master as tgm
import pandas as pd
import stt.google.stt as g_stt
import stt.aws.stt as aws_stt
import stt.kaldi_docker.stt as kaldi_stt
import stt.ibm.stt as ibm_stt
from pydub import AudioSegment
import json
from google.cloud import speech_v1p1beta1 as speech
from importlib import reload


reload(ibm_stt)


google_config = dict(
    language_code="en-AU",
    enable_automatic_punctuation=True,
    enable_word_time_offsets=True,
    enable_word_confidence=True
)
def get_args():
    parser = argparse.ArgumentParser(description='Create kaldi data dir from txtgrid')
    parser.add_argument("sWaveFile", type=str, help='The path to the speech file')

    parser.add_argument("sTxtgridFile", type=str, help='The path to the txtgrid file')

    parser.add_argument("sOutDir", type=str, help='Destination to save kadi data files')

    parser.add_argument("-a", "--append", dest='isAppend', action='store_true', default=False, help='Append to existing files, if not called replace them')

    parser.add_argument("-sid", "--spkr_ID", default='0001', help='Speaker ID')

    parser.add_argument("-rid", "--rcrd_ID", default='0001', help='Record ID')

    parser.add_argument("-p", "--prmpt_tier", default='Prompt', help='Txtgrid tier contains prompts')
    
    return(parser.parse_args())

"""
Merge all consecutive intervals where diff < 2*offset
"""
def get_valid_data(sTxtgridFile, sPromptTier='Prompt', offset=0.0, bMerge=True):#, sSpkrID='0001', sRcrdID='0001', ):
    dTiers = tgm.ParseTxtGrd(sTxtgridFile)
    if bMerge and offset > 0:
        dTiers = tgm.Merge_labels(dTiers, min_sil_dur=2*offset)
        sPromptTier = 'm-{}'.format(sPromptTier)
    lST, lET, lLabels = dTiers[sPromptTier]
    #print(sPromptTier, len(lST))
    data = pd.DataFrame.from_dict({'start_time':lST,'end_time':lET,'label':lLabels})
    data.loc[:,'start_time'] = data.start_time - offset
    data.loc[:,'end_time'] = data.end_time + offset
    data.loc[0,'start_time'] = 0 if data.loc[0,'start_time'] < 0 else data.loc[0,'start_time']
    data_valid = data[(data.label != '') & (data.label != 'sil') & (data.label != '<p:>')]
    return(data_valid)

def kaldi_words_to_dict(response, tierName='kaldi-words', shift_time = 0):
    dTiers = {}
    #dTiers[tierName] = ([],[],[])
    response = response[response.dur > 0]
    response = response.drop_duplicates(subset='startTime', keep='last')
    dTiers[tierName] = [response.startTime.values +shift_time, response.startTime.values+response.dur.values+shift_time, response.symb.values]
    return dTiers


def aws_words_to_dict(response, tierName='aws-words', shift_time = 0):
    dTiers = {}
    dTiers[tierName] = ([],[],[])
    words = response.loc['items'].results
    for word in words:
        word_phrase = word['alternatives'][0]['content']
        try:
            st = float(word['start_time'])
        except KeyError:
            st = -1
        try:
            et = float(word['end_time'])
        except KeyError:
            et = -1
        if st != -1 and et != -1 and st != et:
            dTiers[tierName][2].append(word_phrase)
            dTiers[tierName][0].append(st + shift_time)
            dTiers[tierName][1].append(et + shift_time)
    return dTiers
        

def google_words_to_dict(response, tierName='google-words', shift_time = 0):
    dTiers = {}
    dTiers[tierName] = ([],[],[])
    for result in response.results:
        for word in result.alternatives[0].words:
            word_phrase = word.word
            st = word.start_time.total_seconds()
            et = word.end_time.total_seconds()
            if st != et:
                dTiers[tierName][2].append(word_phrase)
                dTiers[tierName][0].append(st + shift_time)
                dTiers[tierName][1].append(et + shift_time)
    return dTiers


def _process_data(sWaveFile, data, lang = 'en-AU', spkr_ID='0001', rcrd_ID = '0001', out_dir='tmp', asr_engine='google'):
    #Read speech file
    _wav_param, speech_data = tgm.ReadWavFile(sWaveFile)
    data['start_byte'] = (data.start_time * _wav_param.framerate * _wav_param.sampwidth).astype(int)
    data['end_byte'] = (data.end_time * _wav_param.framerate * _wav_param.sampwidth).astype(int)
    if not isdir(out_dir):
        makedirs(out_dir)
    i = -1
    for r in data.iterrows():
        i += 1
        #if i < 20:
        #    continue
        #if i > 20:
        #    break
        shift_time = r[1].start_time
        st_indx = r[1].start_byte
        ed_indx = r[1].end_byte
        label = r[1].label
        st_indx = int(st_indx/2)*2
        ed_indx = int(ed_indx/2)*2
        print(st_indx,ed_indx)
        #Save speech file
        file_name = '{0}_{1}_{2}.wav'.format(spkr_ID, rcrd_ID, r[0])
        file_path = join(out_dir,file_name)
        base_file_path = splitext(file_path)[0]
        txtgrid_file_path_relative = '{0}_{1}_{2}_relative.TextGrid'.format(base_file_path, asr_engine, lang)
        txtgrid_file_path = '{0}_{1}_{2}.TextGrid'.format(base_file_path, asr_engine, lang)
        config_file = '{0}_{1}.config'.format(base_file_path, asr_engine)
        response_file = '{0}_{1}_{2}.json'.format(base_file_path, asr_engine, lang)
        tgm.WriteWaveSegment(speech_data[st_indx-100: ed_indx+100], _wav_param, (ed_indx-st_indx+200)/_wav_param.sampwidth, file_path)
        audio = AudioSegment.from_file(file_path, 'wav')
        audio_duration = audio.duration_seconds
        print(file_name, label)
        if asr_engine == 'google':
            tierName = 'google-words'
            print('send to gcp stt.....')
            config = g_stt.generate_config(lang = lang)#, boost_phrases=[label])
            with open(config_file, 'w') as jf:
                json.dump(config, jf)
            if isfile(response_file):
                print('response file found ...')
                with open(response_file,'r') as jf:
                    phrase = json.load(jf)
                response = speech.RecognizeResponse.from_json(phrase)
            else:
                response = g_stt.stt_single_file(file_path,config=config, overwrite = False)
                with open(response_file, 'w') as jf:
                    json.dump(type(response).to_json(response), jf)
            dTiers = google_words_to_dict(response=response,tierName=tierName, shift_time=shift_time)

        elif asr_engine == 'aws':
            tierName = 'aws-words'
            print('send to aws stt.....')
            response = aws_stt.stt_audio_file(file_path, bucket_name=aws_stt.asr_bucket_name, lang= lang)
            dTiers = aws_words_to_dict(response=response,shift_time=shift_time)
            response.to_json(response_file)
        
        


        tgm.WriteTxtGrdFromDict(txtgrid_file_path_relative, dTiers, shift_time, audio_duration + shift_time, sFilGab='', bReset=False)
        tgm.WriteTxtGrdFromDict(txtgrid_file_path, dTiers, shift_time, audio_duration + shift_time, sFilGab='', bReset=True)
        
    return

def process_data(sWaveFile, data, lang = 'en-AU', spkr_ID='0001', rcrd_ID = '0001', out_dir='tmp', asr_engine='google', forced_upload = False, kaldi_model='model4'):
    #Read speech file
    #_wav_param, speech_data = tgm.ReadWavFile(sWaveFile)
    #data['start_byte'] = (data.start_time * _wav_param.framerate * _wav_param.sampwidth).astype(int)
    #data['end_byte'] = (data.end_time * _wav_param.framerate * _wav_param.sampwidth).astype(int)
    
    if not isdir(out_dir):
        makedirs(out_dir)
    i = -1
    for r in data.iterrows():
        i += 1
        #if i < 20:
        #    continue
        #if i > 20:
        #    break
        start_time = r[1].start_time
        end_time = r[1].end_time
        shift_time = start_time
        label = r[1].label
        #st_indx = int(st_indx/2)*2
        #ed_indx = int(ed_indx/2)*2
        #print(st_indx,ed_indx)
        #Save speech file
        file_name = '{0}_{1}_{2}.wav'.format(spkr_ID, rcrd_ID, r[0])
        file_path = join(out_dir,file_name)
        base_file_path = splitext(file_path)[0]
        txtgrid_file_path_relative = '{0}_{1}_{2}_relative.TextGrid'.format(base_file_path, asr_engine, lang)
        txtgrid_file_path = '{0}_{1}_{2}.TextGrid'.format(base_file_path, asr_engine, lang)
        config_file = '{0}_{1}.config'.format(base_file_path, asr_engine)
        response_file = '{0}_{1}_{2}.json'.format(base_file_path, asr_engine, lang)
        #tgm.WriteWaveSegment(speech_data[st_indx-100: ed_indx+100], _wav_param, (ed_indx-st_indx+200)/_wav_param.sampwidth, file_path)
        tgm.SplitWav(sWaveFile, start_time, end_time, file_path)
        audio = AudioSegment.from_file(file_path, 'wav')
        audio_duration = audio.duration_seconds

        print(file_name, label)
        if asr_engine == 'google':
            tierName = 'google-words'
            print('send to gcp stt.....')
            config = g_stt.generate_config(lang = lang, boost_phrases=[label])
            with open(config_file, 'w') as jf:
                json.dump(config, jf)
            if isfile(response_file):
                print('response file found ...')
                with open(response_file,'r') as jf:
                    phrase = json.load(jf)
                response = speech.RecognizeResponse.from_json(phrase)
            else:
                response = g_stt.stt_single_file(file_path,config=config, overwrite = forced_upload)
                with open(response_file, 'w') as jf:
                    json.dump(type(response).to_json(response), jf)
            dTiers = google_words_to_dict(response=response,tierName=tierName, shift_time=shift_time)

        elif asr_engine == 'aws':
            tierName = 'aws-words'
            print('send to aws stt.....')
            if isfile(response_file):
                response = pd.read_json(response_file)
            else:
                response = aws_stt.stt_audio_file(file_path, bucket_name=aws_stt.asr_bucket_name, lang= lang)
                response.to_json(response_file)
            dTiers = aws_words_to_dict(response=response,shift_time=shift_time)
            
        elif asr_engine == 'kaldi':
            tierName = 'kaldi-words'
            print('send to kaldi stt.....')
            response, response_ph = kaldi_stt.stt_audio_file(file_path, model=kaldi_model)
            dTiers = kaldi_words_to_dict(response=response,shift_time=shift_time)
            if response_ph==None:
                print('no phone align')
            else:
                dTiers_ph = kaldi_words_to_dict(response=response_ph, tierName='kaldi-phones', shift_time=shift_time)
                dTiers.update(dTiers_ph)
            response.to_json(response_file)

        tgm.WriteTxtGrdFromDict(txtgrid_file_path_relative, dTiers, shift_time, audio_duration + shift_time, sFilGab='', bReset=False)
        tgm.WriteTxtGrdFromDict(txtgrid_file_path, dTiers, shift_time, audio_duration + shift_time, sFilGab='', bReset=True)
    results_pattern = join(out_dir,'{0}_{1}_*_{2}_{3}_relative.TextGrid'.format(spkr_ID, rcrd_ID, asr_engine, lang))
    lTxtGrids = glob.glob(results_pattern)
    assert len(lTxtGrids) > 0, 'No matching files'
    sOutTxtGrd = join(out_dir,'{0}_{1}_{2}_{3}_concat.TextGrid'.format(spkr_ID, rcrd_ID, asr_engine, lang))
    dConcatTiers = tgm.ConcatTxtGrids(lTxtGrids)
    tgm.WriteTxtGrdFromDict(sOutTxtGrd, dConcatTiers, 0, dConcatTiers['{0}-words'.format(asr_engine)][1][-1]) 
    return

def evaluate_asr(sTxtGridRef, sRefTierName, lang = 'en-AU', spkr_ID='0001', rcrd_ID = '0001', result_dir='tmp', asr_engine='google'):
    results_pattern = join(result_dir,'{0}_{1}_*_{2}_{3}_relative.TextGrid'.format(spkr_ID, rcrd_ID, asr_engine, lang))
    lTxtGrids = glob.glob(results_pattern)
    assert len(lTxtGrids) > 0, 'No matching files'
    sOutTxtGrd = join(result_dir,'{0}_{1}_{2}_{3}_concat.TextGrid'.format(spkr_ID, rcrd_ID, asr_engine, lang))
    dConcatTiers = tgm.ConcatTxtGrids(lTxtGrids)
    tgm.WriteTxtGrdFromDict(sOutTxtGrd, dConcatTiers, 0, dConcatTiers['{0}-words'.format(asr_engine)][1][-1])
    dRefTier = tgm.ParseTxtGrd(sTxtGridRef)[sRefTierName]
    dASRTier = dConcatTiers['{0}-words'.format(asr_engine)]
    sEvalFile = join(result_dir,'{0}_{1}_{2}_{3}.eval'.format(spkr_ID, rcrd_ID, asr_engine, lang))
    nWords = 0
    nMiss = 0
    StDev = []
    EndDev = []
    with open(sEvalFile, 'w') as fEv:
        for s, e, w in zip(*dRefTier):
            w = w.upper()
            if w in ('UNK','SIL','EXCLUDE'):
                continue
            nWords += 1
            Found = False
            for s_a,e_a,w_a in zip(*dASRTier):
                w_a = w_a.upper()
                if w_a in ('UNK','SIL','sil') or w_a != w:
                    continue
                if s >= s_a and s<= e_a:
                    if e >= e_a:
                        print(w_a, s, e, s_a, e_a, 1, sep=',', file=fEv)
                    else:
                        print(w_a, s, e, s_a, e_a, 4, sep=',', file=fEv)
                    Found = True
                elif e>= s_a and e<= e_a:
                    print(w_a, s, e, s_a, e_a, 3, sep=',', file=fEv)
                    Found = True
                if s_a >= e or Found:
                    break
            if Found:
                stDev = abs(s-s_a)
                edDev = abs(e-e_a)
                if stDev != 0 or edDev != 0:
                    StDev.append(stDev)
                    EndDev.append(edDev)
            else:
                print(w, s, e, -1, sep=',', file=fEv)
                nMiss += 1

    return nWords, nMiss, StDev, EndDev


def ibm_diarization(sWaveFile):
    result = ibm_stt.stt_audio_file(sWaveFile)
    return result

def main():
    args = get_args()

    data = get_valid_data(args.sWaveFile, args.sTxtgridFile, args.prmpt_tier)
    
    return


if __name__ == '__main__':
    main()

