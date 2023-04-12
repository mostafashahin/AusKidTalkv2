import pandas as pd
import tools.txtgrid_master.TextGrid_Master as tgm
from os.path import isfile, basename, splitext
from typing import Tuple, List,NewType
import numpy as np
from sys import argv
from pathlib import Path



offset = 0
agree_th = 0.85
low_agree_pass_threshold = 0.05
foundtarget_pass_threshold = 0.99

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

#OLD VERSION#
def _computeEval_(df_prompt: pd.DataFrame, df_GT: pd.DataFrame, df_AN: pd.DataFrame) -> Tuple[int, int, int, int, List[Tuple[pd.DataFrame,pd.DataFrame]]]:
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


def _computeEval(df_prompt: pd.DataFrame, df_GT: pd.DataFrame, df_AN: pd.DataFrame) -> Tuple[int, int, int, int, List[Tuple[pd.DataFrame,pd.DataFrame]]]:
    GT_AN_hit = AN_Extra = AN_Miss = Miss_Target = 0
    hit_intervals = []
    miss_intervals = []
    for index, row in df_prompt.iterrows():
        ST, ET, target = row[1:]
        if target != 'count_eggs_in_grass':
            GT_find = df_GT[(df_GT.ST >= ST-offset) & (df_GT.ET <= ET+offset) & (df_GT.labels.str.lower() == target.lower())]
            AN_find = df_AN[(df_AN.ST >= ST-offset) & (df_AN.ET <= ET+offset) & (df_AN.labels.str.lower() == target.lower())]
            overlap = False
            if not GT_find.empty:
                if not AN_find.empty:
                    for AN_index, AN_row in AN_find.iterrows():
                        AN_ST, AN_ET, _ = AN_row[1:]
                        for GT_index, GT_row in GT_find.iterrows():
                            GT_ST, GT_ET, _ = GT_row[1:]
                            if (AN_ST >= GT_ST and AN_ST <= GT_ET) or (AN_ET >= GT_ST and AN_ET <= GT_ET) or (AN_ST <= GT_ST and AN_ET >= GT_ET):
                                overlap =True
                    if overlap:
                        GT_AN_hit += 1
                        hit_intervals.append((GT_find, AN_find))
                    else:
                        AN_Miss += 1
                        miss_intervals.append(GT_find)
                else:
                    AN_Miss += 1
                    miss_intervals.append(GT_find)
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
                        for AN_index, AN_row in AN_find.iterrows():
                            AN_ST, AN_ET, _ = AN_row[1:]
                            for GT_index, GT_row in GT_find.iterrows():
                                GT_ST, GT_ET, _ = GT_row[1:]
                                if (AN_ST >= GT_ST and AN_ST <= GT_ET) or (AN_ET >= GT_ST and AN_ET <= GT_ET) or (AN_ST <= GT_ST and AN_ET >= GT_ET):
                                    overlap =True
                        if overlap:
                            GT_AN_hit += 1
                            hit_intervals.append((GT_find, AN_find))
                        else:
                            AN_Miss += 1
                            miss_intervals.append(GT_find)
                    else:
                        AN_Miss += 1
                        miss_intervals.append(GT_find)
                elif not AN_find.empty:
                    AN_Extra += 1
                    
                else:
                    Miss_Target += 1

    return (GT_AN_hit, AN_Extra, AN_Miss, Miss_Target, hit_intervals, miss_intervals)

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


def create_table(header, data):
    # Calculate the maximum width of each column
    col_widths = [max(len(str(item)) for item in column) for column in zip(header, *data)]

    # Build the table as a string
    table = []
    table.append(" | ".join("{:{}}".format(title, col_widths[i]) for i, title in enumerate(header)))
    table.append("-+-".join("-" * width for width in col_widths))
    for row in data:
        table.append(" | ".join("{:{}}".format(item, col_widths[i]) for i, item in enumerate(row)))
    table_str = "\n".join(table)
    return table_str
        
        
def scoreIt(GT_TxtGridPath, AN_TxtGridPath):
    AN_path = Path(AN_TxtGridPath)
    AN_Name = AN_path.parent.name
    logName = splitext(AN_TxtGridPath)[0] + '.log.csv'
    reportName = splitext(AN_TxtGridPath)[0] + '.report'

    df_GT_prompts = _loadTextGrid(GT_TxtGridPath, 'Prompt')
    df_GT = _loadTextGrid(GT_TxtGridPath, 'Final-wrd')
    df_AN = _loadTextGrid(AN_TxtGridPath, 'hu-wrd')

    GT_AN_hit, AN_Extra, AN_Miss, Miss_Target, hit_intervals, miss_intervals = _computeEval(df_GT_prompts, df_GT, df_AN)

    foundTargetsRatio = round(GT_AN_hit / (GT_AN_hit + AN_Miss),2)

    OR = _computeOverlap(hit_intervals)

    df_OR = pd.DataFrame.from_records(OR, columns=['target', 'GT-Start_Time', 'GT-End_Time', 'AN-Start_Time', 'AN-End_Time', 'Overlap'])
    df_OR = df_OR[df_OR.Overlap != 0.0]
    low_agree = df_OR[df_OR.Overlap < agree_th]
    low_agree_ratio = round(len(low_agree)/len(df_OR),2)
    
    is_pass = True
    if low_agree_ratio > low_agree_pass_threshold or foundTargetsRatio < foundtarget_pass_threshold:
        is_pass = False
    
    
    df_OR.to_csv(logName, index=False)

    OR_mean, OR_median, OR_max, OR_min = [round(x,2) for x in (df_OR.Overlap.mean(), df_OR.Overlap.median(), df_OR.Overlap.max(), df_OR.Overlap.min())]
    ##Write Report
    with open(reportName,'w') as f:
        f.write(f"""{AN_Name}\n\nDetailed Report\n-------------\n
Found Target Ratio = {foundTargetsRatio*100}%
Total Number of Targets = {GT_AN_hit + AN_Miss}
Number of Found Targets = {GT_AN_hit}
Number of Missed Targets = {AN_Miss}
Number of Extra Targets = {AN_Extra}


------------------------
Detail of Missed Targets
------------------------

""")
        header = ['Label','Start-Time','End-Time']
        data = []
        
        for GT in miss_intervals:
            for _, row in GT.iterrows():
                GT_ST, GT_ET, label = row[1:]
                data.append([label,f'{GT_ST:0.2f}',f'{GT_ET:0.2f}'])
        f.write(create_table(header,data))
        f.write('\n')
        
        f.write(f"""
Overlap_ratio_mean = {OR_mean}
Overlap_ratio_median = {OR_median}
Overlap_ratio_max = {OR_max}
Overlap_ratio_min = {OR_min}

Percentage of intervals with Overlap ratio < {agree_th} = {low_agree_ratio*100}%
Number of intervals with Overlap ratio < {agree_th} =  {len(low_agree)}


-------------------------------------------
Detail of intervals of Overlap ratio < {agree_th}
-------------------------------------------


""")
        
        header = ['Label','GT-Start-Time','GT-End-Time','AN-Start-Time','AN-End-Time','OR']
        data = []
        for index, row in low_agree.iterrows():
            target, GT_ST, GT_ET, AN_ST, AN_ET, OR = row
            data.append([target, f'{GT_ST:0.2f}',f'{GT_ET:0.2f}', f'{AN_ST:0.2f}',f'{AN_ET:0.2f}', f'{OR:0.2f}'])
        
        f.write(create_table(header,data))     
        
        
    return (OR_mean, OR_median, OR_max, OR_min, foundTargetsRatio, low_agree_ratio, is_pass)

def main():
    _, GT_TxtGridPath, AN_TxtGridPath = argv

    OR_mean, OR_median, OR_max, OR_min, foundTargetsRatio, low_agree_ratio, is_pass = scoreIt(GT_TxtGridPath, AN_TxtGridPath)
    print(f'{foundTargetsRatio:0.2}, {low_agree_ratio:0.2}, {OR_mean:0.2}, {OR_median:0.2}, {OR_max:0.2}, {OR_min:0.2}, {is_pass}')

    return

if __name__ == '__main__':
    main()
