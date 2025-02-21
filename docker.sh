#!/bin/bash

docker run --platform linux/amd64 -it --user ros --network=host --ipc=host \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY \
-e LIBGL_ALWAYS_SOFTWARE=1 \
-v ./catkin_ws:/catkin_ws lukewarmtemp/rob498:latest