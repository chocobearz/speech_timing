#!/bin/bash

# Define the source and destination directories
src_dir="/mnt/c/Users/ptut0/Documents/speech_timing/basline/data/test"
dest_dir="/mnt/c/Users/ptut0/Documents/speech_timing/basline/data/txt"
done_dir="/mnt/c/Users/ptut0/Documents/speech_timing/basline/data/done"

# Loop until the source directory is empty
while [ "$(ls -A "$src_dir")" ]; do
  # Move the first 20 files to the destination directory
  find "$src_dir" -maxdepth 1 -type f | head -n 20 | xargs -I {} mv {} "$dest_dir" 
  node build_base_voices.js
  find "$dest_dir" -maxdepth 1 -type f | head -n 20 | xargs -I {} mv {} "$done_dir" 
done