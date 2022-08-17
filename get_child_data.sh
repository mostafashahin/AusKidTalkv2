childID=$1

mkdir -p samples/$childID/
scp auskidtalk@149.171.37.243:/volume1/AusKidTalk_Recordings/$childID\\\ *_*_*/$childID\\\ Primary*.wav samples/$childID/

