ChildID=$1
AnnID=$2


python3 script/scoreAnn.py data/${ChildID}/${AnnID}/${ChildID}_task1_GT.TextGrid data/${ChildID}/${AnnID}/${ChildID}_task1_kaldi.TextGrid


