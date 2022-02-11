import sys, time
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from os.path import basename, splitext
from random import randint
from pydub import AudioSegment

asr_bucket_name = 'dpbucketasr'

def convert_to_flac(audio_file, format='wav'):
    audio = AudioSegment.from_file(audio_file, format=format)
    outFileName = splitext(audio_file)[0]+'.flac'
    audio.export(outFileName, format='flac')
    return(outFileName)

def check_key(bucket_name, file_path):
    s3 = boto3.resource('s3')
    Found = False
    try:
        s3.Object(bucket_name, file_path).load()
        Found = True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            Found = False
        else:
            raise
    return Found


def upload_file_to_bucket(bucket_name, file_path, overwrite=False):
    file_basename = basename(file_path)
    Found = check_key(bucket_name, file_basename)
    file_uri = "s3://{0}/{1}".format(bucket_name, file_basename)
    if not Found or overwrite:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path,'dpbucketasr',file_basename)
    return file_uri


def stt_audio_file(file_path, bucket_name, lang='en-AU', overwrite=False, enableDiarization=True, nSpeakers=3):
    flac_file = convert_to_flac(file_path)
    job_uri = upload_file_to_bucket(bucket_name=bucket_name, file_path=flac_file, overwrite=overwrite)
    job_name = basename(file_path) + str(randint(1,10000))
    transcribe = boto3.client('transcribe')
    transcribe.start_transcription_job(TranscriptionJobName=job_name,Media={'MediaFileUri': job_uri},MediaFormat='flac',LanguageCode=lang, Settings = {'ShowSpeakerLabels':enableDiarization, 'MaxSpeakerLabels':nSpeakers})
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(5)
    data = pd.read_json(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
    return data

