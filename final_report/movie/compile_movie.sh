#!/bin/bash

#rm segments/*
#rm movie.mp4
#rm crowds_zara01_short.avi
#
## Time
#INTRO_TIME=5
#ZARA_TIME=35.5
#METHODOLOGY_TIME=32.5
TRAJ1_FRAME_TIME=1.45
#RESULTS_TIME=44.75
#QUAL_RESULTS1_TIME=17
#QUAL_RESULTS2_TIME=22
#TRAJ2_FRAME_TIME=0.5
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
#pdftoppm qual_results1.pdf qual_results1 -png
#ffmpeg -loop 1 -i qual_results1-1.png -c:v libx264 -t $QUAL_RESULTS1_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/qual_results1.mp4
#
#pdftoppm qual_results2.pdf qual_results2 -png
#ffmpeg -loop 1 -i qual_results2-1.png -c:v libx264 -t $QUAL_RESULTS2_TIME -pix_fmt yuv420p -vf scale=1280:720 segments/qual_results2.mp4
#
## Filler
#ffmpeg -fflags +genpts -i crowds_zara01.avi -c copy -t $ZARA_TIME crowds_zara01_short.avi
#ffmpeg -i crowds_zara01_short.avi -vf "pad=width=1280:height=720:x=280:y=72:color=black" segments/zara.mp4
#
sed "s/REPLACE/duration $TRAJ1_FRAME_TIME/g" filelist.txt > filelist_mod.txt
ffmpeg -f concat -i filelist_mod.txt -vf fps=10 -pix_fmt yuv420p segments/trajectories.mp4
ffmpeg -i segments/trajectories.mp4 -vf "pad=width=1280:height=720:x=280:y=72:color=black" segments/trajectories_pad.mp4
#
#sed "s/REPLACE/duration $TRAJ2_FRAME_TIME/g" filelist2.txt > filelist2_mod.txt
#ffmpeg -f concat -i filelist2_mod.txt -vf fps=10 -pix_fmt yuv420p segments/trajectories2.mp4
#ffmpeg -i segments/trajectories2.mp4 -vf "pad=width=1280:height=720:x=280:y=72:color=black" segments/trajectories2_pad.mp4


# Combine All
mkvmerge -o movie.mp4 \
segments/introduction.mp4 \
+ segments/zara.mp4 \
+ segments/methodology.mp4 \
+ segments/trajectories_pad.mp4 \
+ segments/results.mp4 \
+ segments/qual_results1.mp4 \
+ segments/qual_results2.mp4 \
+ segments/trajectories2_pad.mp4 \

# Add Audio
ffmpeg -i movie.mp4 -i audio.mp3 -c copy -map 0:v:0 -map 1:a:0 final_movie.mp4
