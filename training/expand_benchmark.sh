#!/bin/bash
# Nearly double the benchmark's camera clips. §29: 12 clips resolves a ~5-point
# change (which is why §17's swap was unambiguous) and cannot resolve the
# 1.4-point changes every experiment since has produced. 23 clips cuts the
# detectable effect by ~28%.
#
# Deliberately diverse and all previously unused: sport, snow, flowers, farm,
# city, and four more conference clips. The 2160p versions of existing content
# are skipped -- same scenes, so they add no independent information.
OUT=/tank/webvsr/clips_busy
L=/tank/webvsr/expand_bench.log
B=https://media.xiph.org/video/derf/y4m
for name in snow_mnt_1080p speed_bag_1080p station2_1080p25 sunflower_1080p25 \
            touchdown_pass_1080p tractor_1080p25 west_wind_easy_1080p \
            vidyo1_720p_60fps vidyo3_720p_60fps vidyo4_720p_60fps \
            KristenAndSara_1280x720_60; do
  dst=$OUT/$name.mp4
  [ -s "$dst" ] && continue
  ffmpeg -y -loglevel error -threads 2 -i "$B/$name.y4m" \
    -frames:v 40 -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p \
    "$dst" 2>> $L
  [ -s "$dst" ] && echo "[$(date +%H:%M)] OK $name" >> $L \
                || { echo "[$(date +%H:%M)] FAILED $name" >> $L; rm -f "$dst"; }
done
echo "[$(date +%H:%M)] EXPAND_COMPLETE $(ls $OUT | wc -l) clips total" >> $L
