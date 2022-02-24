import sys
import boto3
import time
import pandas as pd
from os.path import basename
from random import randint

AccessKeyID = 'AKIA2KAPVALBZS7AVX6J'
SecretAccessKey = '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'
sp_file = sys.argv[1]
BaseFileName = basename(sp_file)
bucket_name = 'dpbucketasr'
lang = 'en-US'

s3_client = boto3.client('s3')

s3_client.upload_file(sp_file,'dpbucketasr',BaseFileName)
job_uri = "s3://dpbucketasr/{0}".format(BaseFileName)

job_name = BaseFileName + str(randint(1,10000))

transcribe = boto3.client('transcribe')
transcribe.start_transcription_job(TranscriptionJobName=job_name,Media={'MediaFileUri': job_uri},MediaFormat='wav',LanguageCode=lang)

while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    print("Not ready yet...")
    time.sleep(5)

data = pd.read_json(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
print(data.results.loc['transcripts'][0]['transcript'])

