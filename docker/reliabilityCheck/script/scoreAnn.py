import pandas as pd
import tools.txtgrid_master.TextGrid_Master as tgm
from os.path import isfile, basename, splitext
from typing import Tuple, List,NewType
import numpy as np
from sys import argv



offset = 0
Time = NewType('Time', float)

def _loadTextGrid(path: str, targetTier: str) -> pd.DataFrame:
    if not isfile(path):
        print(f'File {path} not found')
        return None
    dTiers = tgm.ParseTxtGrd(path)
    df = pd.DataFrame.from_dict({'ST':dTiers[targetTier][0],'ET':dTiers[targetTier][1],'labels':dTiers[targetTier][2]})
    df = df[~df.labels.isin(['','sil'])].reset_index()
    if df.empty:
        print(f'Tier {targetTier} in {path} is empty')
        return None
    return df


def _computeEval(df_prompt: pd.DataFrame, df_GT: pd.DataFrame, df_AN: pd.DataFrame) -> Tuple[int, int, int, int, List[Tuple[pd.DataFrame,pd.DataFrame]]]:
    GT_AN_hit = AN_Extra = AN_Miss = Miss_Target = 0
    hit_intervals = []
    for index, row in df_prompt.iterrows():
        ST, ET, target = row[1:]
        if target != 'count_eggs_in_grass':
            GT_find = df_GT[(df_GT.ST >= ST-offset) & (df_GT.ET <= ET+offset) & (df_GT.labels.str.lower() == target.lower())]
            AN_find = df_AN[(df_AN.ST >= ST-offset) & (df_AN.ET <= ET+offset) & (df_AN.labels.str.lower() == target.lower())]
            if not GT_find.empty:
                if not AN_find.empty:
                    GT_AN_hit += 1
                    hit_intervals.append((GT_find, AN_find))
                else:                    
                    AN_Miss += 1
            elif not AN_find.empty:
                AN_Extra += 1
            else:                
                Miss_Target += 1
        else:
            targets_c = ['one','two','three','four','five','six','seven','eight','nine','ten']
            for target in targets_c:
                GT_find = df_GT[(df_GT.ST >= ST-offset) & (df_GT.ET <= ET+offset) & (df_GT.labels.str.lower() == target.lower())]
                AN_find = df_AN[(df_AN.ST >= ST-offset) & (df_AN.ET <= ET+offset) & (df_AN.labels.str.lower() == target.lower())]
                if not GT_find.empty:
                    if not AN_find.empty:
                        GT_AN_hit += 1
                        hit_intervals.append((GT_find, AN_find))
                    else:                        
                        AN_Miss += 1
                elif not AN_find.empty:
                    AN_Extra += 1
                else:
                    Miss_Target += 1
    
    return (GT_AN_hit, AN_Extra, AN_Miss, Miss_Target, hit_intervals)


def _computeOverlap(hit_intervals: List[Tuple[pd.DataFrame,pd.DataFrame]]) -> List[Tuple[str, Time, Time, Time, Time, float]]:
    Overlap_intervals = []
    for GT, AN in hit_intervals:
        for index, row in GT.iterrows():
            GT_ST, GT_ET, label = row[1:]
            for index, row in AN.iterrows():
                AN_ST, AN_ET, _ = row[1:]
                AN_dur = AN_ET - AN_ST
                GT_dur = GT_ET - GT_ST
                if AN_ET <= GT_ST or AN_ST >= GT_ET:
                    shared = 0
                elif GT_ST >= AN_ST:
                    if GT_ET <= AN_ET:
                        shared = GT_dur
                    else:
                        shared = AN_ET - GT_ST
                else:
                    if GT_ET <= AN_ET:
                        shared = GT_ET - AN_ST
                    else:
                        shared = AN_dur
                OR = shared/(GT_dur + AN_dur - shared)
                Overlap_intervals.append((label, GT_ST, GT_ET, AN_ST, AN_ET, OR))
    return Overlap_intervals


def main():
    _, GT_TxtGridPath, AN_TxtGridPath = argv
    
    logName = splitext(AN_TxtGridPath)[0]+'.log.csv'
    
    df_GT_prompts = _loadTextGrid(GT_TxtGridPath, 'Prompt')
    df_GT = _loadTextGrid(GT_TxtGridPath, 'Final-wrd')
    df_AN = _loadTextGrid(AN_TxtGridPath, 'hu-wrd')
    
    GT_AN_hit, AN_Extra, AN_Miss, Miss_Target, hit_intervals = _computeEval(df_GT_prompts, df_GT, df_AN)
    
    foundTargetsRatio = GT_AN_hit/(GT_AN_hit+AN_Miss)
    
    OR = _computeOverlap(hit_intervals)
    
    
    df_OR = pd.DataFrame.from_records(OR, columns=['target', 'GT-Start_Time', 'GT-End_Time', 'AN-Start_Time', 'AN-End_Time', 'Overlap'])
    df_OR = df_OR[df_OR.Overlap !=0.0]
    
    df_OR.to_csv(logName, index=False)
    
    OR_mean, OR_median, OR_max, OR_min = df_OR.Overlap.mean(), df_OR.Overlap.median(), df_OR.Overlap.max(), df_OR.Overlap.min()
    
    print(f'{foundTargetsRatio:0.2}, {OR_mean:0.2}, {OR_median:0.2}, {OR_max:0.2}, {OR_min:0.2}')
    
    return

if __name__ == '__main__':
    main()
    