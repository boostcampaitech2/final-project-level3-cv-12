#!/bin/sh

docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ google_api:0.8 /bin/bash init.sh
docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ sure:0.3 /bin/bash /home/myamya/project/init.sh
docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ sketch:0.4 /bin/bash /home/myamya/project/init.sh

#docker run -it -v /home/myamya/project/image_files/:/home/myamya/project/image_files/ sketch:0.4 /bin/bash




