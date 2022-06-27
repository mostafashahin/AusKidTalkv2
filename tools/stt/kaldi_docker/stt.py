from subprocess import check_output
from io import StringIO
import pandas as pd

ctmNames = ['fileName', 'channel', 'startTime', 'dur', 'symb', 'conf']

def stt_audio_file(fileName, imageName='kaldi-word', suffix=''):
    with open(fileName,'r') as fIN:
        data = check_output(['docker','run','-i',imageName,'bash',f'decode{suffix}.sh'], stdin=fIN)
        print(' '.join(['docker','run','-i',imageName,'bash',f'decode{suffix}.sh']))
        response = pd.read_csv(StringIO(data.decode('ascii')), delim_whitespace=True, names=ctmNames)
    return response, None
