#!/bin/sh
# Makes the intro's files from a source video (any format FFmpeg reads; 1280x720):
#   thorhax_intro.mpg  MPEG-1 video only, quality 3 (about 46 dB PSNR against the source), no B-frames
#   thorhax_intro.pcm  the sound as raw 48 kHz stereo signed 16-bit little-endian PCM
# Usage: app/thorhax_intro/make_intro.sh <video>
set -eu
src="$1"
dir=$(dirname "$0")
ffmpeg -v error -y -i "$src" -an -vf scale=1280:720 -c:v mpeg1video -q:v 3 -bf 0 -f mpeg "$dir/thorhax_intro.mpg"
ffmpeg -v error -y -i "$src" -vn -ac 2 -ar 48000 -f s16le "$dir/thorhax_intro.pcm"
ls -l "$dir/thorhax_intro.mpg" "$dir/thorhax_intro.pcm"
