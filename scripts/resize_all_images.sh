#!/bin/bash

# args: input_dir
INPUT_DIR=$1
OUTPUT_DIR=$2

echo "copying images..."
cp -r $INPUT_DIR $OUTPUT_DIR
cd $OUTPUT_DIR
echo "resizing images..."
find . -name '*.png' -execdir mogrify -resize $3x$4! {} \;
