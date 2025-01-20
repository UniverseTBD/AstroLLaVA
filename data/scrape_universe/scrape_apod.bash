#!/bin/bash

# This script scrapes the Astronomy Picture of the Day (APOD) website via its official API
# https://api.nasa.gov/#apod
# STDIN $1 is your API key

CURRENT_YEAR=$(date +%Y)

for YEAR in $(seq 1995 $CURRENT_YEAR); do
   echo "Scraping APOD for $YEAR"
   if [ $YEAR -eq $CURRENT_YEAR ]; then
       curl "https://api.nasa.gov/planetary/apod?api_key=$1&start_date=$YEAR-06-16" > APODs/${YEAR}_APOD.json
   else
       curl "https://api.nasa.gov/planetary/apod?api_key=$1&start_date=$YEAR-06-16&end_date=$(( $YEAR + 1 ))-06-16" > APODs/${YEAR}_APOD.json
   fi
done

jq --slurp 'add' APODs/* > APODs/all.json
TOTAL=$(jq 'length' APODs/all.json)
echo "$TOTAL APODs gotten"
