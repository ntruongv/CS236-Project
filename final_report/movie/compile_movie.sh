#!/bin/bash

#rm segments/*
#rm movie.mp4
#rm crowds_zara01_short.avi
#
## Time
#INTRO_TIME=10
#ZARA_TIME=10
#METHODOLOGY_TIME=10
#RESULTS_TIME=10
#TRANSITION_TIME=0.5
#
## Introduction
#pdftoppm main.pdf main -png
#ffmpeg -loop 1 -i main-1.png -c:v libx264 -t $INTRO_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/introduction.mp4
#
## Related Work
#
## Methodology
#pdftoppm methodology.pdf methodology -png
#ffmpeg -loop 1 -i methodology-1.png -c:v libx264 -t $METHODOLOGY_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/methodology.mp4
#
## Experiments
#pdftoppm results.pdf results -png
#ffmpeg -loop 1 -i results-1.png -c:v libx264 -t $RESULTS_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/results.mp4
#
## Conclusion
#
## Filler
#ffmpeg -fflags +genpts -i crowds_zara01.avi -c copy -t $ZARA_TIME crowds_zara01_short.avi
#ffmpeg -i crowds_zara01_short.avi -vf "pad=width=1280:height=720:x=280:y=72:color=black" segments/zara.mp4
#
#pdftoppm black.pdf transition -png
#ffmpeg -loop 1 -i transition-1.png -c:v libx264 -t $TRANSITION_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/transition.mp4

# Combine All
mkvmerge -o movie.mp4 \
segments/introduction.mp4 \
+ segments/zara.mp4 \
+ segments/results.mp4
