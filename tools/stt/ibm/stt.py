import sys, time
import pandas as pd
from os.path import basename, splitext, dirname, join
from random import randint
from pydub import AudioSegment
import json
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

url='https://api.au-syd.speech-to-text.watson.cloud.ibm.com'
api_key="AMfX-GrTrD_1oevhMnpShpT4WMOoJniEgYeF1Jz8q9vK"
authenticator = IAMAuthenticator(apikey=api_key)

def convert_to_flac(audio_file, format='wav'):
    audio = AudioSegment.from_file(audio_file, format=format)
    outFileName = splitext(audio_file)[0]+'.flac'
    audio.export(outFileName, format='flac')
    return(outFileName)

def stt_audio_file(file_path,lang='en-AU', enableDiarization=True, model_str='BroadbandModel'):
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(service_url=url)
    flac_file = convert_to_flac(file_path)
    model=''.join([lang,'_',model_str])
    print(model,flac_file)
    with open(flac_file,'rb') as audio_file:
        recognition_job = speech_to_text.create_job(
            audio=audio_file,
            content_type='audio/flac',
            inactivity_timeout=-1,
            model=model,
            speaker_labels=enableDiarization).get_result()
    job_id = recognition_job['id']
    while True:
        status = speech_to_text.check_job(job_id).get_result()
        if status['status'] in ['completed','failed']:
            break
        print("Not ready yet...")
        time.sleep(10)
    result = status['results'] if status['status']=='completed' else status['details'][0]['error']
    return result
    
def stt_audio_file_wav_(file_path,lang='en-AU', enableDiarization=True, model_str='BroadbandModel'):
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(service_url=url)
    #flac_file = convert_to_flac(file_path)
    model=''.join([lang,'_',model_str])
    print(model,file_path)
    with open(file_path,'rb') as audio_file:
        recognition_job = speech_to_text.create_job(
            audio=audio_file,
            content_type='audio/wav',
            inactivity_timeout=-1,
            model=model,
            speaker_labels=enableDiarization).get_result()
    job_id = recognition_job['id']
    while True:
        status = speech_to_text.check_job(job_id).get_result()
        if status['status'] in ['completed','failed']:
            break
        print("Not ready yet...")
        time.sleep(10)
    result = status['results'] if status['status']=='completed' else status['details'][0]['error']
    return result


def stt_audio_file_wav(file_path, format='wav', lang='en-AU', enableDiarization=True, model_str='BroadbandModel'):
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(service_url=url)
    #flac_file = convert_to_flac(file_path)
    model=''.join([lang,'_',model_str])
    print(model,file_path)
    audio = AudioSegment.from_file(file_path, format=format)
    audio = audio.set_frame_rate(16000)
    #with open(file_path,'rb') as audio_file:
    recognition_job = speech_to_text.create_job(
            audio=audio.raw_data,
            content_type='audio/l16;rate=16000',
            inactivity_timeout=-1,
            model=model,
            speaker_labels=enableDiarization).get_result()
    job_id = recognition_job['id']
    while True:
        status = speech_to_text.check_job(job_id).get_result()
        if status['status'] in ['completed','failed']:
            break
        print("Not ready yet...")
        time.sleep(10)
    result = status['results'] if status['status']=='completed' else status['details'][0]['error']
    return result
