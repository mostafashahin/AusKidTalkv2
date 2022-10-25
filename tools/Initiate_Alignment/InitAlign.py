from collections import namedtuple, defaultdict
import sys
import pandas as pd
import numpy as np
import wave, struct
import logging
import configparser, argparse
from os.path import isfile, join, isdir, splitext, basename
from os import makedirs
from joblib import dump, load
from scipy.signal import find_peaks
from tqdm import tqdm
import mysql.connector

sys.path.insert(0,'tools')
import pyAudioAnalysis.ShortTermFeatures as sF
import txtgrid_master.TextGrid_Master as txtgrd

#Configure Logger
logger = logging.getLogger('InitAlign')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('InitAlign.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


#MySQL database connection configuration
#UserName = 'unsw'
#Password = 'UNSWspeech'
#HostIP = '184.168.98.156'
#DatabaseName = 'auskidtalk_prod'

#UserName = 'mostafa'
#Password = 'Hggih;fv2881'
#HostIP = 'localhost'
#DatabaseName = 'auskidstalk'


#date time columns names for tables
dDateTimeColNames = {'task_start_end_times' : ('task1_start_time',
                                               'task1_end_time',
                                               'task1_start_time_2',
                                               'task1_end_time_2',
                                               'task2_start_time',
                                               'task2_end_time',
                                               'task3_start_time',
                                               'task3_end_time',
                                               'task4_start_time',
                                               'task4_end_time',
                                               'task5_start_time',
                                               'task5_end_time'),

        'experiment' : ('answer_time',
                        'task1_audio_cue_offset',
                        'audio_cue_onset',
                        'task2_sentence_offset',
                        'task1_retry1_timestamp',
                        'task1_retry2_timestamp')}


"""TimeStamp CSV Columns
0 - id: child id, dtype 'int'
1 - task_id: task id, dtype 'int'
2 - word_id: prompt id, dtype 'int'
3 - answer_value: rate of child answer, dtype 'int'
4 - answer_time: timestamp where RA press eval, dtype 'timestamp'
5 - task1_attempt_count: number of times prompt repeated, dtype 'int'
6 - task1_audio_cue_offset: timestamp where audio instruction ends, dtype 'timestamp'
7 - audio_cue_onset: timestamp where audio instruction starts, dtype 'timestamp'
8 - ....
"""
"""TaskTimeStamp CSV columns
0 - child_id: child id, dtype 'int'
1 - ra_id:
2 - task1_start_time, dtype 'timestamp'
3 - task1_end_time, dtype 'timestamp'
4 - task2_start_time, dtype 'timestamp'
5 - task2_end_time, dtype 'timestamp'
6 - task3_start_time, dtype 'timestamp'
7 - task3_end_time, dtype 'timestamp'
8 - task4_start_time, dtype 'timestamp'
9 - task4_end_time, dtype 'timestamp'
10 - task5_start_time, dtype 'timestamp'
11 - task5_end_time, dtype 'timestamp'

"""
lTasks = ['task1','task2','task3','task4','task5']
tTaskTimes = namedtuple('TaskTimes',lTasks)
tPrompt = namedtuple('Prompt',['taskID','wordID','word','answerTime','cueOnset','cueOffset','retry1','retry2'])
offset = 2 # seconds added to the end time of each task

def get_args():
    parser = argparse.ArgumentParser(description='Create TextGrid files for each task from timestamps')
    parser.add_argument("iChildID", type=int, help='The children ID')

    parser.add_argument("sWaveFile", type=str, help='The path to the primary mic wav file')

    parser.add_argument("sOutDir", type=str, help='Destination to save textgrid files')

    parser.add_argument("--config_File", type=str, dest='sConfigFile', help='The path to the config file contains parameters for beep detection', default='beep.ini')
    
    parser.add_argument("--database_Name", type=str, dest='sDatabaseName', help='Name of the database. Note that this will overwrite the value in the config file')

    return(parser.parse_args())
    

def GetBeepTimes(sWavFile, nReadFrames = 10, nFramDur = 0.02, zcTh = 0.2, srTh = 0.2, BeepDur = 1, p = 0.8):
    fWav = wave.open(sWavFile,'rb')
    if not isfile(sWavFile):
        raise Exception("Wave file {} not exist".format(sWavFile))
    fr = fWav.getframerate()
    nFrameSamples = int(nFramDur * fr)
    nReadSamples = nReadFrames * nFrameSamples


    nSamples = fWav.getnframes()
    nFrames = int(nSamples/nFrameSamples)
    num_fft = int(nFrameSamples / 2)

    vZC = np.zeros((nFrames+1),dtype=int)
    vSR = np.zeros((nFrames+1),dtype=int)
    
    indx = 0
    while fWav.tell() <= nSamples-nReadSamples:
        data = fWav.readframes(nReadSamples)
        data = list(struct.iter_unpack('h',data))
        #Normalization and remove dc-shift if any
        data = np.double(data)
        data = data / (2.0 ** 15)
        dc_offset = data.mean()
        maximum = (np.abs(data)).max()
        data = (data - dc_offset) / maximum
        
        for iFr in range(nReadFrames):
            Fram_data = data[iFr * nFrameSamples : (iFr+1) * nFrameSamples,0]
            vZC[indx] = int(sF.zero_crossing_rate(Fram_data) > zcTh)
            
            # get fft magnitude
            fft_magnitude = abs(sF.fft(Fram_data))

            # normalize fft
            fft_magnitude = fft_magnitude[0:num_fft]
            fft_magnitude = fft_magnitude / len(fft_magnitude)

            vSR[indx] = int(sF.spectral_rolloff(fft_magnitude,0.9) > srTh)

            indx += 1
    fWav.close()

    BeepnFrames = int(BeepDur/nFramDur)
    sum_zc = np.sum(vZC[:BeepnFrames])
    sum_sr = np.sum(vSR[:BeepnFrames])
    vSum_zc = np.zeros(vZC.shape[0]-BeepnFrames,dtype=int)
    vSum_sr = np.zeros(vZC.shape[0]-BeepnFrames,dtype=int)

    for i in range(vZC.shape[0]-BeepnFrames):
        sum_zc = sum_zc - vZC[i] + vZC[i+BeepnFrames]
        sum_sr = sum_sr - vSR[i] + vSR[i+BeepnFrames]
        vSum_zc[i] = sum_zc
        vSum_sr[i] = sum_sr


    mask_zc = (vSum_zc > p*BeepnFrames).astype(int)
    mask_sr = (vSum_sr > p*BeepnFrames).astype(int)

    mask_zc[0] = mask_zc[-1] = 0
    mask_sr[0] = mask_sr[-1] = 0


    dif_zc = mask_zc - np.roll(mask_zc,1)
    dif_sr = mask_sr - np.roll(mask_sr,1)

    BeepTimes_zc = np.where(dif_zc == 1)[0]*nFramDur
    BeepTimes_sr = np.where(dif_sr == 1)[0]*nFramDur

    return dif_zc, dif_sr, BeepTimes_zc, BeepTimes_sr

#TODO: Speed up beep detection use only MFCC
def GetBeepTimesML(sConfFile, sWavFile, iThrshld=98, fBeepDur = 1):

    #Set Default Values
    sModelFile = ''
    Context = (-2,-1,0,1,2)
    fFrameRate = 0.01
    fWindowSize = 0.02
    bUseDelta = False
    sFeatureType = 'STF'


    #Load Values from ini file
    if not isfile(sConfFile):
        raise Exception('Config file {0} is not exist'.format(sConfFile))
    config = configparser.ConfigParser()
    config.read(sConfFile)
    try:
        Flags = config['FLAGS']
    except KeyError:
        logger.error('Config File {0} must contains section [FLAGS]'.format(sConfFile))
        raise RuntimeError('Config file format error')
    
    if 'Model' not in Flags:
        logger.error('Please set Model parameter in the config file {0}'.format(sConfFile))
        raise RuntimeError('Config file format error')
    else:
        sModelFile = Flags['Model']

    if 'Context' in Flags:
        Context = tuple([int(i) for i in Flags['Context'].split(',')])
    if 'FrameRate' in Flags:
        fFrameRate = Flags.getfloat('FrameRate')
    if 'WindowSize' in Flags:
        fWindowSize = Flags.getfloat('WindowSize')
    if 'UseDelta' in Flags:
        bUseDelta = Flags.getboolean('UseDelta')
    if 'FeatureType' in Flags:
        sFeatureType = Flags['FeatureType']
        


    if not sModelFile:
        raise Exception('Please set Model parameter in the config file {0}'.format(sConfFile))

    #Load Model
    if not isfile(sModelFile):
        raise Exception('Model file {0} is not exist'.format(sModelFile))

    clf = load(sModelFile)

    nChunkSize = 1000 #Number of frames to read each time

    #Get number of padded rows for context
    nPostPad = max(Context)
    nPrePad = abs(min(Context))

    #Beep Detection
    if not isfile(sModelFile):
        raise Exception('Wave file {0} is not exist'.format(sWavFile))

    
    with wave.open(sWavFile) as fWav:
        iSampleRate = fWav.getframerate()
        nSamples = fWav.getnframes()
        assert fWav.getsampwidth() == 2, 'Only 16 bit resolution supported, Please convert the file'

        nBeepFrames = int(fBeepDur/fFrameRate)

        nFrames = int(nSamples / (fFrameRate * iSampleRate))
        logger.info("Processing file {0} contains {1} frames".format(sWavFile,nFrames))

        aBeepMask = np.zeros(nFrames,dtype=int)

        nStepSamples = int(fFrameRate*iSampleRate)
        nWindowSamples = int(fWindowSize*iSampleRate)
        nOverLabSamples = nWindowSamples - nStepSamples

        i = 0
        with tqdm(total=nFrames) as pbar:
            while fWav.tell() < nSamples-nWindowSamples:
                #print(nChunkSize,fStepSize,iFrameRate)
                data = fWav.readframes(int(nChunkSize*nStepSamples)+nWindowSamples)
                data = [ x[0] for x in struct.iter_unpack('h',data)]
                data = np.asarray(data)
                aFeatures, lFeatures_names = sF.feature_extraction(data,iSampleRate,nWindowSamples,nStepSamples,deltas=bUseDelta)

                aFeatures = aFeatures.T
                #Handle context
                aPostPad = np.zeros((nPostPad,aFeatures.shape[1]))
                aPrePad = np.zeros((nPrePad,aFeatures.shape[1]))

                aFeatures_pad = np.r_[aPrePad,aFeatures,aPostPad]

                aShiftVer = [np.roll(aFeatures_pad, i, axis=0) for i in Context[::-1]] #To handle context generate multiple shifted versions, this method faster but consume memory 

                aFeatures = np.concatenate(aShiftVer,axis=1)[0+nPrePad:-nPostPad]


                X = aFeatures

                y_pred = clf.predict(X)
                
                aBeepMask[i:i+y_pred.shape[0]] = y_pred
                
                i = i+y_pred.shape[0]

                logger.debug('done {} frames out of {} frames'.format(i,nFrames))
                
                
                
                fWav.setpos(fWav.tell() - nOverLabSamples)

                #print(fWav.tell(),nSamples)
                
                pbar.update(y_pred.shape[0])
        
        suma=np.sum(aBeepMask[:nBeepFrames])
        vSum = np.zeros(aBeepMask.shape[0]-nBeepFrames)
        for i in range(aBeepMask.shape[0]-nBeepFrames):
            vSum[i] = suma
            suma = suma - aBeepMask[i] + aBeepMask[i+nBeepFrames]

        lPeaks = find_peaks(vSum,height=iThrshld)[0]

        lBeepTimes = lPeaks * fFrameRate
        
        logger.info('File {0}: {1} beeps detected at {2}'.format(sWavFile,len(lBeepTimes),lBeepTimes))

    return lBeepTimes

#To Handle multiple timestamps in same column
def date_time_list(x,fn):
    lx = x.split(',')
    lx = [i for i in lx if i != '0']
    lx = [None] if not lx else lx
    lx_t = pd.to_datetime(lx, format='%Y-%m-%d', errors='coerce')
    return fn(lx_t)

def GetTimeStampsSQL(iChildID, sConfigFile,sDatabaseName=None):
    
    UserName = 'unsw'
    Password = 'UNSWspeech'
    HostIP = '184.168.98.156'
    DatabaseName = 'auskidtalk_prod'
    
    #Load Values from ini file
    if not isfile(sConfigFile):
        raise Exception('Config file {0} is not exist'.format(sConfigFile))
    config = configparser.ConfigParser()
    config.read(sConfigFile)
    try:
        sql_conf = config['SQL']
    except KeyError:
        logger.error('Config File {0} must contains section [SQL]'.format(sConfigFile))
        raise RuntimeError('Config file format error')
    
    if 'UserName' in sql_conf:
        UserName = sql_conf['UserName']
    if 'Password' in sql_conf:
        Password = sql_conf['Password']
    if 'HostIP' in sql_conf:
        HostIP = sql_conf['HostIP']
    if 'DatabaseName' in sql_conf:
        DatabaseName = sql_conf['DatabaseName']
    
    #Overwrite with arg
    if sDatabaseName:
        DatabaseName = sDatabaseName
        
    try:
        connector = mysql.connector.connect(user=UserName, password=Password,
                              host=HostIP,
                              database=DatabaseName,buffered=True)
    except Exception as err:
        logger.error('MySQL connection failed with error \n{0}'.format(err))
        raise RuntimeError('Database connection failuer')


    cursor = connector.cursor(buffered=True, dictionary=True)
    #Check existance of the words, task_start_end_times, experiment tables

    query = ("SHOW TABLES")
    cursor.execute(query)
    results = cursor.fetchall()
    if not results:
        logger.error('No Tables in the database {0}'.format(DatabaseName))
        raise RuntimeError('Database empty')
    #print(results)
    lTableNames = [list(i.values())[0] for i in results]
    for table in ('words', 'task_start_end_times', 'experiment'):
        if table not in lTableNames:
            logger.error('Missing table {0} in database {1}'.format(table, DatabaseName))
            raise RuntimeError('Missing table in database')


    #Read task_start_end_times table
    query = ("SELECT * FROM task_start_end_times WHERE child_id={0}".format(iChildID))
    cursor.execute(query)
    results = cursor.fetchall()

    if len(results) == 0:
        logger.error('child {}: No data for the child in the task_start_end_times table in database {}'.format(iChildID,DatabaseName))
        raise RuntimeError("Data missing in task_start_end_times table for child {}, check log for more info".format(iChildID))
    
    if len(results) > 1:
        logger.error('child {}: multiple records for the child in the task_start_end_times table in database {}'.format(iChildID,DatabaseName))
        raise RuntimeError("Multiple records for the child in the task_start_end_times table for child {}, check log for more info".format(iChildID))

    pdChild_Task = pd.DataFrame.from_dict(results)
    
    #Convert string to datetime for timestamp columns
    #for sColName in dDateTimeColNames['task_start_end_times']:
        #pdChild_Task[sColName] = pd.to_datetime(pdChild_Task[sColName], format='%Y-%m-%d', errors='coerce')
    for sColName in dDateTimeColNames['task_start_end_times']:
        if 'start' in sColName:
            pdChild_Task[sColName] = pdChild_Task[sColName].apply(date_time_list,args=[min])
        elif 'end' in sColName:
            pdChild_Task[sColName] = pdChild_Task[sColName].apply(date_time_list,args=[max])

    child_task_tstamps = pdChild_Task.iloc[-1]
    
    if pd.isnull(child_task_tstamps.task1_start_time):
        logger.error('child {}: No time stamp for the start of task 1, Reference time can\'t set'.format(iChildID))

        raise RuntimeError("Error in task timestamp file for child {}".format(iChildID))


    RefTime = child_task_tstamps.task1_start_time.timestamp()


    #Load experiment table
    query = ("SELECT * FROM experiment WHERE id={0}".format(iChildID))
    cursor.execute(query)
    results = cursor.fetchall()

    if len(results) == 0:
        logger.error('child {}: No data for the child in the experiment table in database {}'.format(iChildID,DatabaseName))
        raise RuntimeError("Data missing in experiment table for child {}, check log for more info".format(iChildID))
    
    pdChild = pd.DataFrame.from_dict(results)
    
    #Convert string to datetime for timestamp columns
    for sColName in dDateTimeColNames['experiment']:
        pdChild[sColName] = pd.to_datetime(pdChild[sColName])


    #Load words ID table
    query = ("SELECT * FROM words")
    cursor.execute(query)
    results = cursor.fetchall()

    if len(results) == 0:
        logger.error('child {}: Words table is empty')
        raise RuntimeError("words table is empty")
    
    pdWordIDs = pd.DataFrame.from_dict(results).set_index('word_id')
    dWordIDs = pdWordIDs.to_dict()['name']


    dTaskPrompts = defaultdict(list)
    lTaskTimes = []


    for i,sTaskID in enumerate(lTasks):
        iTaskID = i+1
        if sTaskID == 'task1':
            fTaskST,fTaskET_p1, fTaskST_p2, fTaskET = child_task_tstamps.loc['{0}_start_time'.format(sTaskID):'{0}_end_time_2'.format(sTaskID)]

        else:
            fTaskST, fTaskET = child_task_tstamps.loc['{0}_start_time'.format(sTaskID):'{0}_end_time'.format(sTaskID)]
        #print(fTaskST,fTaskET,iTaskID)
        

        if pd.isnull(fTaskST):
            logger.warning('child {0}: No start timestamp for task {1}'.format(iChildID,sTaskID))
            fTaskST = -1
            #lTaskTimes.append((-1,-1))
        else:
            fTaskST = fTaskST.timestamp() - RefTime

        if pd.isnull(fTaskET):
            logger.warning('child {0}: No end timestamp for task {1}'.format(iChildID,sTaskID))
            fTaskET = -1
        else:
            fTaskET = fTaskET.timestamp() - RefTime
        if sTaskID == 'task1':
            if pd.isnull(fTaskET_p1):
                fTaskET_p1 = -1
                #lTaskTimes.append((-1,-1))
            else:
                fTaskET_p1 = fTaskET_p1.timestamp() - RefTime

            if pd.isnull(fTaskST_p2):
                fTaskST_p2 = -1
            else:
                fTaskST_p2 = fTaskST_p2.timestamp() - RefTime
        
        
        logger.info('child {0} task {1} starttime {2} endtime {3}'.format(iChildID,sTaskID,fTaskST ,fTaskET))
        

        if sTaskID=='task1':
            lTaskTimes.append((fTaskST,fTaskET_p1, fTaskST_p2, fTaskET))
        else:
            lTaskTimes.append((fTaskST ,fTaskET))

        pdTask = pdChild[pdChild.task_id==iTaskID] ##CHANGE if COL CHANGED
        
        if pdTask.empty:
            logger.warning('child {}: No data of task {} in database {}, task will be skipped'.format(iChildID, sTaskID, DatabaseName))
            continue
        
        for r in pdTask.iterrows():
            #TODO handle any nonexist field
            data = r[1]
            iWordID = data.word_id
            if pd.isnull(iWordID) or iWordID not in dWordIDs:
                logger.warning('child {}: word id {} of task {} either null or not exist in word-mappingfile word set to NULL'.format(iChildID, str(iWordID), sTaskID))
                sWord = 'NULL'
            else:
                sWord = dWordIDs[iWordID]
            
            answerTime = data.answer_time
            if pd.isnull(answerTime):
                logger.warning('child {}: answer timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                answerTime = -1
            else:
                answerTime = answerTime.timestamp() - RefTime
       
            cueOnset = data.audio_cue_onset
            if pd.isnull(cueOnset):
                logger.warning('child {}: cueOnset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOnset = -1
            else:
                cueOnset = cueOnset.timestamp() - RefTime

            cueOffset = data.task1_audio_cue_offset
            if pd.isnull(cueOffset):
                logger.warning('child {}: cueOffset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOffset = -1
            else:
                cueOffset = cueOffset.timestamp() - RefTime
            
            n_attempts = data.task1_attempt_count
            if n_attempts ==0:
                retry1 = -1
                retry2 = -1
            else:
                retry1 = data.task1_retry1_timestamp
                if pd.isnull(retry1):
                    logger.warning('child {}: task1_retry1 timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                    retry1 = -1
                else:
                    retry1 = retry1.timestamp() - RefTime

                retry2 = data.task1_retry2_timestamp
                if pd.isnull(retry2):
                    logger.warning('child {}: task1_retry2 timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                    retry2 = -1
                else:
                    retry2 = retry2.timestamp() - RefTime

            prompt = tPrompt(iTaskID, iWordID, sWord, answerTime, cueOnset, cueOffset, retry1, retry2)
            
            dTaskPrompts[iTaskID].append(prompt)

    tTasks = tTaskTimes(*lTaskTimes)
    #TODO return 4 times for task1
    return tTasks, dTaskPrompts


def ParseTStampCSV(sTStampFile, sTaskTStampFile, iChildID, sWordIDsFile):
    
    #Check files existance
    if not isfile(sTStampFile):
        raise Exception("child {}: timestamp file not exist".format(iChildID,sTStampFile))
    if not isfile(sTaskTStampFile):
        raise Exception("child {}: task timestamp file not exist".format(iChildID,sTaskTStampFile))
    if not isfile(sWordIDsFile):
        raise Exception("child {}: word mapping file not exist".format(iChildID,sWordIDsFile))


    #Load the prompt mapping file
    pdWordIDs = pd.read_csv(sWordIDsFile,index_col=0)
    dWordIDs = pdWordIDs.to_dict()['name']

    #Load the task timestamps file
    data_task = pd.read_csv(sTaskTStampFile,parse_dates=list(range(2,12)))
    pdChild_Task = data_task[data_task.child_id == iChildID]
    if pdChild_Task.empty:
        logger.error('child {}: No data for the child in the task timestamps file {}'.format(iChildID,sTaskTStampFile))
        raise RuntimeError("Data missing in task timestamp file for child {}, check log for more info".format(iChildID))
    
    if pdChild_Task.shape[0] > 1:
        logger.warning('child {}: more than one line in the task timestamps file {}, only one line expected\nonly last line considered'.format(iChildID,sTaskTStampFile))

    child_task_tstamps = pdChild_Task.iloc[-1]
    if pd.isnull(child_task_tstamps.task1_start_time):
        logger.error('child {}: No time stamp for the start of task 1 in file {}, Reference time can\'t set'.format(iChildID,sTaskTStampFile))

        raise RuntimeError("Error in task timestamp file for child {}".format(iChildID))

    RefTime = child_task_tstamps.task1_start_time.timestamp()
    
    #Load prompt timestamps file
    data = pd.read_csv(sTStampFile,parse_dates=[4,6,7])
    pdChild = data[data.id==iChildID]
    
    if pdChild.empty:
        logger.error('child {}: No data for the child in the prompt timestamps file {}'.format(iChildID, sTStampFile))
        raise RuntimeError("Data missing in task timestamp file for child {}, check log for more info".format(iChildID))

    dTaskPrompts = defaultdict(list)
    lTaskTimes = []

    for i,sTaskID in enumerate(lTasks):
        iTaskID = i+1
        #TODO Use column names
        fTaskST,fTaskET = child_task_tstamps[2*i+2:2*i+4] #First two columns for the child_id and ra_id
        #print(fTaskST,fTaskET,iTaskID)

        if pd.isnull(fTaskST):
            logger.warning('child {0}: No start timestamp for task {1} in file {2}'.format(iChildID,sTaskID,sTaskTStampFile))
            fTaskST = -1
            #lTaskTimes.append((-1,-1))
        else:
            fTaskST = fTaskST.timestamp() - RefTime

        if pd.isnull(fTaskET):
            logger.warning('child {0}: No end timestamp for task {1} in file {2}'.format(iChildID,sTaskID,sTaskTStampFile))
            fTaskET = -1
        else:
            fTaskET = fTaskET.timestamp() - RefTime


        lTaskTimes.append((fTaskST ,fTaskET))

        pdTask = pdChild[pdChild.task_id==iTaskID] ##CHANGE if COL CHANGED
        
        if pdTask.empty:
            logger.warning('child {}: No data of task {} in the prompt timestamps file {}, task will be skipped'.format(iChildID, sTaskID, sTStampFile))
            continue
        
        for r in pdTask.iterrows():
            #TODO handle any nonexist field
            data = r[1]
            iWordID = data.word_id #TODO handle if ID not exist
            if pd.isnull(iWordID) or iWordID not in dWordIDs:
                logger.warning('child {}: word id {} of task {} either null or not exist in word-mappingfile word set to NULL'.format(iChildID, str(iWordID), sTaskID))
                sWord = 'NULL'
            else:
                sWord = dWordIDs[iWordID]
            
            answerTime = data.answer_time
            if pd.isnull(answerTime):
                logger.warning('child {}: answer timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                answerTime = -1
            else:
                answerTime = answerTime.timestamp() - RefTime
       
            cueOnset = data.audio_cue_onset
            if pd.isnull(cueOnset):
                logger.warning('child {}: cueOnset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOnset = -1
            else:
                cueOnset = cueOnset.timestamp() - RefTime

            cueOffset = data.task1_audio_cue_offset
            if pd.isnull(cueOffset):
                logger.warning('child {}: cueOffset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOffset = -1
            else:
                cueOffset = cueOffset.timestamp() - RefTime

            prompt = tPrompt(iTaskID, iWordID, sWord, answerTime, cueOnset, cueOffset)
            
            dTaskPrompts[iTaskID].append(prompt)

    tTasks = tTaskTimes(*lTaskTimes)
    return tTasks, dTaskPrompts


def _GetOffsetTime(tTasks, lBeepTimes):
    #Get number of tasks
    nTasks = len(tTasks)
    nBeepTimeStamps = []
    #TODO for task1 there are 2 beeps use them for more accurate offset time estimation
    for fTaskST, fTaskET in tTasks:
        nBeepTimeStamps.append(fTaskST) if fTaskST != -1 else print('')
    lDifTimes = []
    for i in range(len(lBeepTimes)):
        for j in range(len(nBeepTimeStamps)):
            lDifTimes.append(abs(lBeepTimes[i]-nBeepTimeStamps[j]))
    
    lDifTimes = np.asarray(lDifTimes)
    lDifTimes.sort()
    
    iEqualDiss = np.where(np.diff(lDifTimes,n=1,axis=0) < 1 )[0]
    
    if iEqualDiss.size == 0:
        logger.error('Failed to verify beep times')
        fOffsetTime=-1
    else:
        fOffsetTime = np.mean(lDifTimes[iEqualDiss])
    
    return fOffsetTime

def GetOffsetTime(tTasks, lBeepTimes):
    startTimes = [i[0] for i in tTasks]
    startTimes.append(tTasks[0][2]) #Adding starttime of the second part of task1 fTaskST_p2
    startTimes.sort()
    diff_ts = []
    nTasks = len(startTimes)
    for i in range(nTasks):
        for j in range(i+1,nTasks):
            diff_ts.append((i,j,startTimes[j]-startTimes[i]))
    diff_beeps = []
    nBeeps = len(lBeepTimes)
    for i in range(nBeeps):
        for j in range(i+1,nBeeps):
            diff_beeps.append((i,j,lBeepTimes[j]-lBeepTimes[i]))
    offsets = []
    for ib,jb,diffb in diff_beeps:
        for it, jt, difft in diff_ts:
            if abs(diffb - difft) < 2:
                offsets.append(lBeepTimes[ib]-startTimes[it])
                offsets.append(lBeepTimes[jb]-startTimes[jt])
    if not offsets:
        logger.error('Failed to verify beep times')
        fOffsetTime = -555555 #Cause it could be negative to some extend
    else:
        fOffsetTime = np.mean(offsets)
    
    return fOffsetTime

#def Segmentor(sConfigFile, sWavFile, sTimeStampCSV, sTaskTStampCSV, iChildID, sWordIDsFile, sOutDir):
def Segmentor(sConfigFile, sWavFile, iChildID, sOutDir,sDatabaseName=None):

    #TODO get child ID from wav file
    #TODO verify naming convention of file
    #print('Segmentor')
    #Load Wav File (session)
    logger.info("Child {}: Start Segmentation.....")
    if not isfile(sWavFile):
        logger.error("Child {}: session speech File {} not exist".format(iChildID,sWavFile))
        raise Exception("Child {}: session speech File {} not exist".format(iChildID,sWavFile))
    
    sWavFileBasename = splitext(basename(sWavFile))[0]

    if not isdir(sOutDir):
        makedirs(sOutDir)
     
    logger.info('Child {}: Start Processing File {}'.format(iChildID,sWavFile))
    
    logger.info('Child {}: Getting timestamps'.format(iChildID))
    try:
        #tTasks, dPrompts = ParseTStampCSV(sTimeStampCSV, sTaskTStampCSV, iChildID, sWordIDsFile)
        tTasks, dPrompts = GetTimeStampsSQL(iChildID, sConfigFile,sDatabaseName=sDatabaseName)
    except:
        logger.error('Child {}: Error while getting timestamps'.format(iChildID))
        raise Exception("Child {}: Error while getting timestamps".format(iChildID))


    nTasks = len(tTasks)

    logger.info('Child {}: {} tasks timestamps detected'.format(iChildID, nTasks))

    for i in range(nTasks):
        iTaskID = i+1

        if iTaskID in dPrompts:
            logger.info('Child {}: task {} contains {} prompts'.format(iChildID, iTaskID, len(dPrompts[iTaskID])))
        else:
            logger.info('Child {}: task {} contains {} prompts'.format(iChildID, iTaskID, 0))

    logger.info('Child {}: Getting Beep times'.format(iChildID))
    sBeepsFile = splitext(sWavFile)[0]+'.Beeps'
    loadbeeps = False
    if isfile(sBeepsFile):
        try:
            lBeepTimes = np.loadtxt(sBeepsFile)
            loadbeeps = True
        except:
            logger.warning('Child {}: Error while loading beep times'.format(iChildID))
            loadbeeps = False
    if not loadbeeps:
        try:
            lBeepTimes = GetBeepTimesML(sConfigFile, sWavFile)
            np.savetxt(sBeepsFile,lBeepTimes)
        except:
            logger.error('Child {}: Error while detecting beep times'.format(iChildID))
            raise Exception("Child {}: Error while detecting beep times".format(iChildID))

    logger.info('Child {}: {} beeps detected @ times {}'.format(iChildID, len(lBeepTimes),lBeepTimes))
    #lBeepTimes = np.asarray([  10.23,  644.65, 3283.61, 4095.36])
    #lBeepTimes = lBeepTimes / 100.0
    try:
        fOffsetTime = GetOffsetTime(tTasks,lBeepTimes)
        logger.info('Child {}: time offset = {}'.format(iChildID, fOffsetTime))
    except:
        logger.error('Child {}: Error while getting offset time'.format(iChildID))
        raise Exception("Child {}: Error while getting offset time".format(iChildID))


    if fOffsetTime == -555555:
        raise Exception("child {}: session speech File {} not exist".format(iChildID,sWavFile))
     
    logger.info('Child {}: offset time {}'.format(iChildID,fOffsetTime))

    #testWav = '../../Recordings/13_aug_2020/90 3_2_0/90 Primary_15-01.wav'
    _wav_param, RWav = txtgrd.ReadWavFile(sWavFile)
    


    for i in range(nTasks):

        iTaskID = i + 1

        logger.info('Child {}: Annotating task {}'.format(iChildID,iTaskID))

        if len(tTasks[i]) == 4:
            fTaskST,_,_,fTaskET = tTasks[i]
        else:
            fTaskST,fTaskET = tTasks[i]
        fRefTime = fTaskST 
        #Fix missing start and end times of tasks, if start missing use end of previous task, if end time missing use start time of next task
        if fTaskST == -1:
            if i ==0:
                fTaskST = 0
            else:
                fTaskST = tTasks[i-1][1]
        if fTaskET == -1:
            if i == nTasks -1:
                fTaskET = _wav_param.nframes/_wav_param.framerate
            else:
                fTaskET = tTasks[i+1][0]
        
        fTaskST += fOffsetTime
        if fTaskST < 0:
            fTaskST = 0
            fRefTime = -fOffsetTime
        #fTaskST = 0 if fTaskST < 0 else fTaskST
        fTaskET += fOffsetTime
        
        fTaskSF = int(fTaskST*_wav_param.framerate*_wav_param.sampwidth)
        fTaskEF = int(fTaskET*_wav_param.framerate*_wav_param.sampwidth)
        
        #As the sample width is 2 bytes, the start and end positions should be even number
        fTaskSF += (fTaskSF%2)
        fTaskEF += (fTaskEF%2)
        
        nFrams = int((fTaskEF-fTaskSF)/_wav_param.sampwidth)
        
        txtgrd.WriteWaveSegment(RWav[fTaskSF:fTaskEF],_wav_param,nFrams,join(sOutDir,'{}_task{}.wav'.format(iChildID,iTaskID)))
        ETime = nFrams / _wav_param.framerate

        #Generate textgrids
        #TODO Fix tasks 3 & 4
        if iTaskID in [3,4]:
            continue
        dTiers = defaultdict(lambda: [[],[],[]])
        lPrompts = dPrompts[iTaskID]
        for p in lPrompts:
            times = [t for t in p[3:] if t != -1]  #Get the min of all repeats and max of all repeats
            fST, fET, label = min(times), max(times), p.word
            dTiers['Prompt'][0].append(fST-fRefTime)
            dTiers['Prompt'][1].append(fET-fRefTime)
            dTiers['Prompt'][2].append(label)
        #print(dTiers)
        dTiers = txtgrd.SortTxtGridDict(dTiers)
        

        #REMOVE THIS########################################
        with open(join(sOutDir,'{}_task{}.int'.format(sWavFileBasename,iTaskID)),'w') as f:
            for fST, fET, sLabel in zip(*dTiers['Prompt']):
                print(fST, fET, sLabel, file=f)
        ############################################


        #dTiers = txtgrd.FillGapsInTxtGridDict(dTiers)
        #dump(dTiers,'dTier{}.jbl'.format(iTaskID))
        
        txtgrd.WriteTxtGrdFromDict(join(sOutDir,'{}_task{}_prompt.TextGrid'.format(iChildID,iTaskID)),dTiers,0.0,ETime,sFilGab="")


    #Detect the start and end of all beep signal(s)
    #ParseCSV file get tTasks, dTaskPrompts
    #TODO fill interval gaps with empty, this should be done in the writetextgrid function
    """
    _wav_param, RWav = txtgrd.ReadWavFile('Recordings/24_jan_2020/Data/CH001/CH001_1_001
     ...: .wav')
     fRefPos = IA.GetBeepTime()
     fStTime = tTasks.task1[0] + fRefPos
     fEtTime = tTasks.task1[1] + fRefPos
     iFSt = int(fStTime*_wav_param.framerate*_wav_param.sampwidth)
     iFEt = int(fEtTime*_wav_param.framerate*_wav_param.sampwidth)
     nFrams = int((iFEt-iFSt)/_wav_param.sampwidth)
     txtgrd.WriteWaveSegment(RWav[iFSt:iFEt],_wav_param,nFrams,'task1.wav')
     dTiers = defaultdict(lambda: [[],[],[]])
     lSt = [t.cueOffset for t in tPrompts[1]]
     lEt = [t.answerTime for t in tPrompts[1]]
     lLabel = [t.word for t in tPrompts[1]]
     dTiers['Prompt']=[lSt,lEt,lLabel]
     txtgrd.WriteTxtGrdFromDict('task1.txtgrid',dTiers,0.0,lEt[-1])

    """

def main():
    args = get_args()

    iChildID, sWaveFile, sOutDir = args.iChildID, args.sWaveFile, args.sOutDir

    sConfigFile = args.sConfigFile #if 'sConfigFile' in args else 'beep.ini'
    sDatabaseName = args.sDatabaseName
        
    
    #try:
    #    print('trying....')
    Segmentor(sConfigFile, sWaveFile, iChildID, sOutDir, sDatabaseName=sDatabaseName)
    #except:
    #    logger.error('Segmentor failed')


if __name__ == '__main__':
    main()
