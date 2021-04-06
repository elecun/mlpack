
#!/bin/bash

if [ -z "$1" ]
  then
    echo "No Argument"
    echo "Usage : ./avi_gen.sh <max id number>"
    exit 1
fi

for ((i=0; i<$1; i++));
do
  ffmpeg -framerate 25 -i ./images/fsr_matrix/$i/relax_crop_%d.jpg ./images/fsr_matrix/$i/relax_crop.avi
  ffmpeg -framerate 25 -i ./images/fsr_matrix/$i/standard_crop_%d.jpg ./images/fsr_matrix/$i/standard_crop.avi
done