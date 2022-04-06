#!/usr/bin/env bash

# Copyright 2020  Mostafa Shahin (UNSW)

# This script run #TODO

PYTHON=/usr/bin/python3


DIR=samples/test2/
OUT_DIR=$DIR/annotate1
stage=0

#TODO list directories only
ls $DIR | while read direct
do


    echo "Checking Directory $direct"

    WAV_FILE=`ls  "$DIR/$direct/" | grep Primary`

    [ -z "$WAV_FILE" ] && echo "ERROR: Primary wav file not exist!" && continue

    childID=`echo "$WAV_FILE" | cut -d' ' -f1`

    echo "Child ID $childID"

    #TODO check if childID already processed //DONE

    WAV_FILE_PATH=$DIR/$direct/$WAV_FILE

    LOCAL_OUT_DIR=$OUT_DIR/$childID

    echo "Starting child $childID "


    mkdir -p "$LOCAL_OUT_DIR"

    #Convert to 16 bit
    echo "Converting to 16 bit for beep detection"
    if [ ! -f $LOCAL_OUT_DIR/primary_16b.wav ]; then
        sox "$WAV_FILE_PATH" -c 1 -b 16 "$LOCAL_OUT_DIR/primary_16b.wav" || continue
    else
        echo "Converted file exist of child $childID"
    fi
    #TODO validate converted wav file, same duration of the original one
    #TODO if converted file exist don't convert again
    n=`ls $LOCAL_OUT_DIR/txtgrids/*_task?_prompt.TextGrid | wc -l`
    if [ $n -ne 3 ]; then
        ! $PYTHON tools/Initiate_Alignment/InitAlign.py --config_File scripts/beep.ini $childID "$LOCAL_OUT_DIR/primary_16b.wav" $LOCAL_OUT_DIR/txtgrids && mv InitAlign.log "$LOCAL_OUT_DIR" && continue
        mv InitAlign.log "$LOCAL_OUT_DIR"
    fi
done
