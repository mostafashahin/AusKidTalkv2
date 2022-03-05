from subprocess import check_output
from io import StringIO
import pandas as pd

ctmNames = ['fileName', 'channel', 'startTime', 'dur', 'symb', 'conf']

def stt_audio_file(fileName, imageName='kaldi-word'):
    with open(fileName,'r') as fIN:
        data = check_output(['docker','run','-i',imageName,'bash','decode.sh'], stdin=fIN)
        response = pd.read_csv(StringIO(data.decode('ascii')), delim_whitespace=True, names=ctmNames)
    return response, None