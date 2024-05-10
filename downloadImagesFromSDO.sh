#!/bin/bash
######################################################################
#DownloadImage_Linux.sh
#Author: Kevin M. Addison
#Description: Downloads browse images using start date, end date, channel, and resolution parameters
#Published: 2022-08-04
#Schedule: no schedule
#Input Params: STARTDATE, ENDDATE, CHANNEL, RESOLUTION, DOWNLOAD_PATH
#Date Format: YYYY-MM-DD
#Channels: [0193,0171,0304,0131,0335,0211,0094,1600,1700,HMIB,HMIBC,HMII,HMIIC,HMIIF,HMID]
#Resolutions: [4096,2048,1024,512]
#Commandline Instructions: chmod +x DownloadImage_Linux.sh
#                          ./DownloadImage_Linux.sh 2022-01-01 2022-01-02 0171 512 /PATH/TO/DOWNLOADS
######################################################################

# COMMANDLINE ARGUMENTS
STARTDATE=$1
ENDDATE=$2
CHANNEL=$3
RESOLUTION=$4
DOWNLOAD_PATH=$5

# SDO WEBSITE URL
SDOURL=https://sdo.gsfc.nasa.gov
BROWSEDIR=$SDOURL"/assets/img/browse"

# DOWNLOAD PATH
LOCALDIR=$DOWNLOAD_PATH


let DAYS=(`date +%s -d ${ENDDATE}`-`date +%s -d ${STARTDATE}`)/86400

echo -e "\n\n"
echo "Download Images to local directory"
echo "START DATE: "$STARTDATE
echo "END DATE: "$ENDDATE
echo "CHANNEL: "$CHANNEL
echo "RESOLUTION: "$RESOLUTION
echo "DOWNLOAD PATH: "$LOCALDIR
echo -e "\n"

val=0
for (( i = 1; i <= $DAYS; i++ ))
do
	SUBDIR=$(date +%Y-%m-%d -d "$STARTDATE + $i day")
	NEXTDATEPATH=$(date +%Y/%m/%d -d "$STARTDATE + $i day")
	NEXTDATESTRING=$(date +%Y%m%d -d "$STARTDATE + $i day")
	URL=${BROWSEDIR}/${NEXTDATEPATH}

	FILE_PATTERN="${NEXTDATESTRING}_.+_${RESOLUTION}_${CHANNEL}.jpg"


	# ファイルリストを取得（リダイレクトに従う）
	FILE_LIST=$(curl -L -s $URL | grep -oP '(?<=href=")[^"]+' | grep -P "$FILE_PATTERN")
	echo $FILE_LIST

	# 最も若い時刻のファイルを選択
	YOUNGEST_FILE=$(echo "$FILE_LIST" | sort | head -n 1)

    	# ファイルをダウンロード
    	if [ -n "$YOUNGEST_FILE" ]; then
        	wget -N -q --no-check-certificate $URL/$YOUNGEST_FILE --directory-prefix=$LOCALDIR/$SUBDIR
    	fi
done

echo -e "\n"Script complete: $(date)

