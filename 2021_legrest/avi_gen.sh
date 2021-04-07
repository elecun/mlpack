
#!/bin/bash

if [ -z "$1" ]
  then
    echo "No Argument"
    echo "Usage : ./avi_gen.sh <max id number>"
    exit 1
fi

for ((i=1; i<=$1; i++));
do
  echo "converting..."$i
  ffmpeg -framerate 25 -i ./images/fsr_matrix/$i/relax_crop_%d.jpg ./avi/relax_crop_$i.avi > /dev/null 2>&1
  ffmpeg -framerate 25 -i ./images/fsr_matrix/$i/standard_crop_%d.jpg ./avi/standard_crop_$i.avi > /dev/null 2>&1
done