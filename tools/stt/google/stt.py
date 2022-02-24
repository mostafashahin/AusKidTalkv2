from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud.storage import blob
from pydub import AudioSegment
from os.path import splitext, basename
from google.oauth2 import service_account
json_path="/home/mostafa/root/stt/google/speechtotext-314810-c6c261fa64a9.json"
my_credentials = service_account.Credentials.from_service_account_file(json_path)

#TODO How to switch between acounts
word_time = True
config = dict(
    language_code="en-AU",
    enable_automatic_punctuation=True,
    enable_word_time_offsets=word_time,
    enable_word_confidence=True
)
bucket_name = 'dpsydneynsw_speech'
def generate_config(lang = 'en-AU', punc = False, wrd_time = True, wrd_conf = True, boost_phrases = []):
    config = dict (
        language_code=lang,
        enable_automatic_punctuation=punc,
        enable_word_time_offsets=wrd_time,
        enable_word_confidence=wrd_conf
    )
    if boost_phrases:
        config['speech_contexts'] = [{"phrases":boost_phrases}]
    return config

def stt_file(config, audio):
    client = speech.SpeechClient(credentials=my_credentials)
    response = client.recognize(config=config, audio=audio)
    return(response)

def stt_long_file(config, audio):
    client = speech.SpeechClient(credentials=my_credentials)
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()
    return(response)
    
def get_transcript(response):
    transcript = []
    confidence = []
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript.append(best_alternative.transcript)
        confidence.append(best_alternative.confidence)
    return(transcript,confidence)

def get_trans_with_words(response):
    transcript = []
    confidence = []
    words = []
    words_ST = []
    words_ET = []
    words_conf = []
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript.append(best_alternative.transcript)
        confidence.append(best_alternative.confidence)
        for word in best_alternative.words:
            words.append(word.word)
            words_ST.append(word.start_time.total_seconds())
            words_ET.append(word.end_time.total_seconds())
            words_conf.append(word.confidence)
    return(transcript,confidence,words, words_ST, words_ET, words_conf)
def get_transcript_(response):
    result = response.results[0] #For single portion audio
    best_alternative = result.alternatives[0]
    transcript = best_alternative.transcript
    confidence = best_alternative.confidence
    return(transcript, confidence)

def print_sentences(response):
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript = best_alternative.transcript
        confidence = best_alternative.confidence
        print("-" * 80)
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence:.0%}")
        print_word_offsets(best_alternative)

def print_word_offsets(alternative):
    for word in alternative.words:
        start_s = word.start_time.total_seconds()
        end_s = word.end_time.total_seconds()
        word = word.word
        print(f"{start_s:>7.3f} | {end_s:>7.3f} | {word}")

def convert_to_flac(audio_file, format='wav'):
    audio = AudioSegment.from_file(audio_file, format=format)
    outFileName = splitext(audio_file)[0]+'.flac'
    audio.export(outFileName, format='flac')
    return(outFileName)

def upload_file_to_bucket(bucket_name, file_path, overwrite=False):
    storage_client = storage.Client(credentials=my_credentials)
    bucket = storage_client.bucket(bucket_name=bucket_name)
    file_name = basename(file_path)
    blob = bucket.blob(file_name)
    if not blob.exists() or overwrite:
        blob.upload_from_filename(file_path)
    return('gs://{}/{}'.format(bucket_name,file_name))


def stt_batch(files_path):
    batch_trans = {}
    for wav_file in files_path:
        print('processing {}'.format(wav_file))
        res = stt_single_file(wav_file, config)
        batch_trans[wav_file] = get_transcript(res)
    return(batch_trans)

def stt_single_file(wav_file, config, overwrite = False):
    flac_file = convert_to_flac(wav_file)
    print(flac_file)
    gs_path = upload_file_to_bucket(bucket_name, flac_file, overwrite=overwrite)
    print(gs_path)
    audio = dict(uri=gs_path)
    audio_data = AudioSegment.from_file(wav_file, 'wav')
    if audio_data.duration_seconds <= 60:
        print('process short file')
        res = stt_file(config,audio=audio)
    else:
        print('process long file')
        res = stt_long_file(config,audio=audio)
    return res

def write_dict_to_file(file_path, d):
    with open(file_path,'w') as f:
        for i in d.keys():
            print(i, *d[i], sep='\t', file=f)

