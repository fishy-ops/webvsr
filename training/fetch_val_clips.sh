#!/bin/bash
# Widen the held-out validation set from 4 camera clips to 10. Section 23
# disqualified it on a flicker disagreement with the benchmark, and 4 clips
# against 12 was the most likely explanation -- this tests that.
# Adds video-conference content, which is a large share of real web video and
# is absent from both the benchmark and the first validation set.
OUT=/tank/webvsr/clips_val
mkdir -p $OUT
L=/tank/webvsr/fetch_val.log
B=https://media.xiph.org/video/derf/y4m
for name in 720p50_parkrun_ter 720p50_shields_ter 720p5994_stockholm_ter \
            FourPeople_1280x720_60 Johnny_1280x720_60 rush_field_cuts_1080p; do
  dst=$OUT/$name.mp4
  [ -s "$dst" ] && continue
  echo "[$(date +%H:%M)] fetching $name" >> $L
  ffmpeg -y -loglevel error -threads 2 -i "$B/$name.y4m" \
    -frames:v 32 -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p \
    "$dst" 2>> $L
  [ -s "$dst" ] && echo "[$(date +%H:%M)] OK $name" >> $L || { echo "[$(date +%H:%M)] FAILED $name" >> $L; rm -f "$dst"; }
done
echo "[$(date +%H:%M)] WIDEN_COMPLETE $(ls $OUT | wc -l) clips" >> $L
