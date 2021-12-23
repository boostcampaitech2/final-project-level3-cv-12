python sketch.py

find /home/myamya/project/image_files/cropped_512_image -name "*_out*"|sed -e 'p' -e 's/_out//g' | xargs -n 2 mv
cp /home/myamya/project/image_files/annotationsx4/annotationsx4_file.txt /home/myamya/project/image_files/annotationsx4/annotationsx4_file.json


dir=/home/myamya/project/image_files/sketched1
FOLDER_NAME=$(basename $(dirname $(realpath $0)))
for entry in $dir/*
do
        file="${entry##/*/}"
        python ./sketch_simplification/simplify.py --img /home/myamya/project/image_files/sketched1/$file --out /home/myamya/project/image_files/sketched2/$file --model sketch_gan.t7

done
find /home/myamya/project/image_files/sketched2 -name "*_out*"|sed -e 'p' -e 's/_out//g' | xargs -n 2 mv

python /home/myamya/project/alignment.py
