from os import chdir, getcwd, remove
from os.path import isfile
import pandas as pd
from io import StringIO
from subprocess import check_output
kaldi_path = '/media/Windows/root/kaldi/egs/akt-asr/work'
ctmNames = ['fileName', 'channel', 'startTime', 'dur', 'symb', 'conf']
#ctmPhNames = ['fileName', 'channel', 'startTime', 'dur', 'symb', 'conf']

def stt_audio_file(audio_file, lang='en-US',model='model4'):
    currentDir = getcwd()
    chdir(kaldi_path)
    #print(getcwd())
    #data = check_output(['decode_libri_tedlium.sh', '/home/mostafa/root/AusKidTalk/Data/refData/279_task1_125.wav'])
    #data = check_output(['bash', 'decode_libri_task1_tedlium.sh', audio_file])
    if isfile('phone.ali'): remove('phone.ali')
    #data = check_output(['bash', 'decode_child_task1_tedlium.sh', audio_file])
    data = check_output(['bash', 'decode_child_task1_tedlium_varModel.sh', audio_file, model])
    #data = check_output(['bash', 'decode_au_task1_tedlium.sh', audio_file])
    response = pd.read_csv(StringIO(data.decode('ascii')), delim_whitespace=True, names=ctmNames)
    response_ph=None
    if isfile('phone.ali'):
        response_ph = pd.read_csv('phone.ali', delim_whitespace=True, names=ctmNames)
        response_ph['symb'] = [c.split('_')[0] for c in response_ph.symb.values]
    chdir(currentDir)
    return response, response_ph



