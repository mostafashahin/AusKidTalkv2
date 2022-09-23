from subprocess import check_output
from io import StringIO
import pandas as pd

ctmNames = ['fileName', 'channel', 'startTime', 'dur', 'symb', 'conf']

def stt_audio_file(fileName, imageName='kaldi-word', suffix='', bPhoneme=False):
    print(' '.join(['docker','run','-i',imageName,'bash',f'decode{suffix}.sh']))
    with open(fileName,'r') as fIN:
        data = check_output(['docker','run','-i',imageName,'bash',f'decode{suffix}.sh','--align-ph',str(bPhoneme).lower()], stdin=fIN)
        data = data.decode('ascii').split(';')
        if len(data) == 2:
            data_w, data_p = data
            response_w = pd.read_csv(StringIO(data_w), delim_whitespace=True, names=ctmNames)
            response_p = pd.read_csv(StringIO(data_p), delim_whitespace=True, names=ctmNames)
        else:
            data_w = data[0]
            response_w = pd.read_csv(StringIO(data_w), delim_whitespace=True, names=ctmNames)
            response_p = None
    return response_w, response_p
