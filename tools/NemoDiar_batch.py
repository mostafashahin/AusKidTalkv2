import os
import wget
from omegaconf import OmegaConf
import json
import sys
from os.path import join, basename, exists, splitext
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import pandas as pd
import librosa
import numpy as np
import txtgrid_master.TextGrid_Master as tm


childIDList = sys.argv[1]
taskID = sys.argv[2]
data_dir = sys.argv[3]

#Nemo Config
pretrained_speaker_model = 'titanet_large'
window_length_in_sec = [1.5,1.25,1.0,0.75,0.5]
shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1]
multiscale_weights= [1,1,1,1,1]
pretrained_vad = 'vad_multilingual_marblenet'
msdd_model = 'diar_msdd_telephonic' # Telephonic speaker diarization model

with open(childIDList,'r') as listFile:
    for childID in listFile.read().splitlines():

        wav_file = join(data_dir,childID,'txtgrids',f"{childID}_{taskID}.wav")
        manifest_file = splitext(wav_file)[0]+'_nemo.json'
        txtgrid_file = splitext(wav_file)[0]+'_nemo.TextGrid'
        #manifest_file = join(data_dir,childID,f"{childID}_{taskID}.json")
        #txtgrid_file = join(data_dir,childID,f"{childID}_{taskID}_nemo.TextGrid")
        if exists(txtgrid_file):
            continue

        meta = {
            'audio_filepath': wav_file,
            'offset': 0,
            'duration':None,
            'label': 'infer',
            'text': '-',
            'num_speakers': 3,
            'rttm_filepath': None,
            'uem_filepath' : None
        }

        with open(manifest_file,'w') as fp:
            json.dump(meta,fp)
            fp.write('\n')

        output_dir = join(data_dir, childID,'nemo_outputs')
        MODEL_CONFIG = join(data_dir,'diar_infer_telephonic.yaml')
        if not os.path.exists(MODEL_CONFIG):
            config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
            MODEL_CONFIG = wget.download(config_url,data_dir)

        config = OmegaConf.load(MODEL_CONFIG)
        print(OmegaConf.to_yaml(config))
        config.diarizer.vad.parameters.onset = 0.05
        config.diarizer.manifest_filepath = manifest_file
        config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = window_length_in_sec
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = shift_length_in_sec
        config.diarizer.speaker_embeddings.parameters.multiscale_weights= multiscale_weights
        #config.diarizer.oracle_vad = True # ----> ORACLE VAD 
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.msdd_model.model_path = msdd_model
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7,1.0]
        system_vad_msdd_model = NeuralDiarizer(cfg=config)

        system_vad_msdd_model.diarize()
        dur_in_secs = librosa.get_duration(filename=wav_file)
        output_rttm = join(output_dir,f"pred_rttms/{childID}_{taskID}.rttm")
        df_rttm = pd.read_csv(output_rttm, delim_whitespace=True, keep_default_na=False, names=['Type','File_ID','Channel_ID','Turn_Onset','Turn_Duration','Orth','Speaker_Type','Speaker_Name','Conf_Score','lokhd'])
        df_rttm['End_Time'] = df_rttm['Turn_Onset']+df_rttm['Turn_Duration']

        tiersDict = {}
        for spk in df_rttm['Speaker_Name'].unique():
            print(spk)
            tierName = spk
            df_rttm_spk = df_rttm[df_rttm.Speaker_Name==spk]
            stTime, endTime, labels = [a.squeeze() for a in np.split(df_rttm_spk[['Turn_Onset','End_Time','Orth']].values,3,1)]
            tiersDict[tierName] = (list(stTime), list(endTime), list(labels))

        tm.WriteTxtGrdFromDict(txtgrid_file,tiersDict,0,dur_in_secs,sFilGab='')
