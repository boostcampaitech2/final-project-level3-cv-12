python sketch.py

dir=/home/myamya/project/image_files/sketched1
FOLDER_NAME=$(basename $(dirname $(realpath $0)))
for entry in $dir/*
do
        file="${entry##/*/}"
        python ./sketch_simplification/simplify.py --img /home/myamya/project/image_files/sketched1/$file --out /home/myamya/project/image_files/sketched2/$file --model sketch_gan.t7

done

