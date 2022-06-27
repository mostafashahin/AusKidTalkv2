import sys, os
sys.path.insert(0,os.path.abspath('tools/'))

import shutil
import txtgrid_master.TextGrid_Master as tgm
import librosa
import soundfile as sf
import comm_asr_from_txtgrid as cm
from importlib import reload
from collections import defaultdict

#TODO replace with argparse
dir = sys.argv[1]
childID = sys.argv[2]
taskID = sys.argv[3]
mapper_file = sys.argv[4]

lang = 'en-AU'
asr_engine='kaldi'
offset=0.5 #Amount used to relax the interval boudries of ibm Textgrid

tmp_dir = os.path.join(dir,'tmp','kaldi')
if not os.path.isdir(dir):
    #print("{} is not exist".format(dir))
    sys.exit("{} is not exist".format(dir))

#tmp_dir = os.path.join(dir,'tmp')
if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)

#TODO check wav file and ibm textgrid are exist
ibm_txtgrid  = os.path.join(dir,'{}_{}_ibm.TextGrid'.format(childID, taskID))
prompt_txtgrid = os.path.join(dir,'{}_{}_prompt.TextGrid'.format(childID, taskID))
wav_file = os.path.join(dir,'{}_{}.wav'.format(childID, taskID))

#Get number of detected speakers by ibm
dTiers = tgm.ParseTxtGrd(ibm_txtgrid)
speakers_tiers = dTiers.keys()

data_prompt = cm.get_valid_data(prompt_txtgrid) #Get non-sil intervals from prompt TextGrid

data_ibm = {}
for spk in speakers_tiers:
    data_ibm[spk] = cm.get_valid_data(ibm_txtgrid, sPromptTier=spk,offset=offset,bMerge=True)


nOverlaps = defaultdict(lambda :0)
for r in data_prompt.iterrows():
    bAdd = True
    for spk in speakers_tiers:
        df = data_ibm[spk]
        crnt_overlap =  df[((df.start_time > r[1].start_time) & (df.start_time < r[1].end_time)) |
                 ((df.start_time < r[1].start_time) & (df.end_time > r[1].start_time))].shape[0]
        nOverlaps[spk] += crnt_overlap
        if crnt_overlap > 0:
            bAdd = False
    if bAdd:
        for spk in speakers_tiers:
            data_ibm[spk].loc[-1] = r[1]
            data_ibm[spk].index = data_ibm[spk].index+1
for spk in speakers_tiers:
    data_ibm[spk].sort_values('start_time',inplace=True)

for spk in speakers_tiers:
    print('processing {}'.format(spk))
    df_data = data_ibm[spk]
    resDir = os.path.join(tmp_dir,spk)
    cm.process_data(sWaveFile=wav_file,data=df_data, lang= lang, spkr_ID=childID, rcrd_ID=taskID, out_dir=resDir, asr_engine=asr_engine, forced_upload=False, kaldi_suffix=taskID)


textgrid_files = []
aSelectedTiers = []
aMapper = []
if os.path.isfile(mapper_file):
    aMapper = [('kaldi-words',mapper_file)]*len(speakers_tiers)
dest_file = os.path.join(dir,'{0}_{1}_{2}.TextGrid'.format(childID, taskID, asr_engine))
for spk in speakers_tiers:
    textgrid_files.append(os.path.join(tmp_dir,spk,'{0}_{1}_{2}_{3}_concat.TextGrid'.format(childID, taskID, asr_engine, lang)))
    aSelectedTiers.append({'kaldi-words':'{}-kaldi-words'.format(spk)})
tgm.MergeTxtGrids(textgrid_files,dest_file,sWavFile=wav_file, aSlctdTiers=aSelectedTiers,aMapper=aMapper)

