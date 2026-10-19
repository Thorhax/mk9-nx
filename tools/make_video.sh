#!/bin/bash
# Turns a raw recording (sa2_video_s, app/src/sa2_capture.cpp) into an MP4 and a contact sheet of frames.
# Same format and usage as the banjo-nx tool.
# Usage: tools/make_video.sh <captures/video_Ns.rgba> [frames in the sheet, default 16] [step between them, default 15]
set -eu
RAW=$1; N=${2:-16}; STEP=${3:-15}
FFMPEG=${FFMPEG:-/c/ffmpeg/ffmpeg.exe}
# The .txt is written on Windows: drop the carriage return before reading it.
read W H FPS COUNT < <(tr -d '\r' < "${RAW%.rgba}.txt")
OUT="${RAW%.rgba}"
"$FFMPEG" -loglevel error -y -f rawvideo -pix_fmt rgb0 -s "${W}x${H}" -r "$FPS" -i "$RAW" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$OUT.mp4"
COLS=4; ROWS=$(( (N + COLS - 1) / COLS ))
# Frame numbers on the sheet need a font file (drawtext without one crashes some Windows ffmpeg builds); without
# drawtext the sheet is made plain.
"$FFMPEG" -loglevel error -y -f rawvideo -pix_fmt rgb0 -s "${W}x${H}" -r "$FPS" -i "$RAW" \
  -vf "select='not(mod(n\,$STEP))',scale=480:-1,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='%{n}':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6,tile=${COLS}x${ROWS}" \
  -frames:v 1 "$OUT.sheet.png" 2>/dev/null || \
"$FFMPEG" -loglevel error -y -f rawvideo -pix_fmt rgb0 -s "${W}x${H}" -r "$FPS" -i "$RAW" \
  -vf "select='not(mod(n\,$STEP))',scale=480:-1,tile=${COLS}x${ROWS}" -frames:v 1 "$OUT.sheet.png"
echo "$OUT.mp4 ($COUNT frames at $FPS fps), sheet $OUT.sheet.png"
