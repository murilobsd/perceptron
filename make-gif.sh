#!/bin/sh

set -xe

# cargo run
ffmpeg -y -i data/weights-%03d.ppm data/demo.mp4
ffmpeg -y -i data/demo.mp4 -vf palettegen data/palette.png
ffmpeg -y -i data/demo.mp4 -i data/palette.png -filter_complex paletteuse demo.gif
