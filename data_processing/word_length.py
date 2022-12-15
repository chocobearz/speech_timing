import os
import align
import pandas as pd
import glob
import numpy as np

'''
to run this code you will need to setup gentle aligner : https://github.com/lowerquality/gentle
use updated align script, and gentle folder included in this repo
to replace the one included in gentle aligner's original repo
'''


audioDirectory = '/localhome/ptuttosi/Documents/GAN/speech_timing/audio/azure_base'
alignerDirectory = '/localhome/ptuttosi/Documents/GAN/speech_timing/data_processing/gentle/'

scripts = {
  "IEO": "It's eleven o'clock",
  "TIE": "That is exactly what happened",
  "IOM": "I'm on my way to the meeting",
  "IWW": "I wonder what this is about",
  "TAI": "The airplane is almost full",
  "MTI": "Maybe tomorrow it will be cold",
  "IWL": "I would like a new alarm clock",
  "ITH": "I think I have a doctor's appointment",
  "DFA": "Don't forget a jacket",
  "ITS": "I think I've seen this before",
  "TSI": "The surface is slick",
  "WSI": "We'll stop in a couple of minutes"
}

dataframe = pd.DataFrame({'filename' : [], 'base_word_lengths': [],'pause_lengths': [], 'neutral_relative_word_lengths': []})

neutral_dataframe = pd.DataFrame({'id':[], 'script': [], 'word_lengths': []})

# begin with neutral voices in order to create relative word lengths for all other files
neutral_files = glob.glob(audioDirectory+"*NEU*")
for neutral_file in neutral_files:
  filename = neutral_file.split("/")[-1].split('.')[0]
  print(filename)
  # only one key in filename guarenteed
  for key in scripts:
    if key in neutral_file:
      script = scripts[key]
      break
  alignment = align.align(neutral_file, script)
  new_row = {'filename': filename, 'base_word_lengths': [], 'pause_lengths': [], 'neutral_relative_word_lengths': 'NA' }
  neutral_row = {'id': [], 'word_lengths': []}
  # if any words fail alignment consider it a corrupted file
  if all(word.case == "success" for word in alignment):
    for i, word in enumerate(alignment):
      new_row['base_word_lengths'].append(word.duration)
      neutral_row['id'] = filename.split('_', 1)[0]
      neutral_row['word_lengths'] = new_row['base_word_lengths']
      neutral_row['script'] = filename.split('_')[1]
      # pauses are the difference between the end of one word and the start of the next
      if i+1 < len(alignment):
        new_row['pause_lengths'].append(alignment[i+1].start - word.end)
  else:
    new_row['base_word_lengths'] = "NA"
    new_row['pause_lengths'] = "NA"
    neutral_row['base_word_lengths'] = "NA"
    neutral_row['pause_lengths'] = "NA"
  neutral_dataframe = pd.concat([neutral_dataframe, pd.Series(neutral_row).to_frame().T], ignore_index=True)
  dataframe = pd.concat([dataframe, pd.Series(new_row).to_frame().T], ignore_index=True)

for root, dirs, files in os.walk(audioDirectory):
  for filename in files:
    print(filename)
    if "NEU" not in filename:
      # only one key in filename guarenteed
      for key in scripts:
        if key in filename:
          script = scripts[key]
          break
      alignment = align.align(os.path.join(root, filename), script)
      new_row = {'filename': filename.split('.', 1)[0], 'base_word_lengths': [], 'pause_lengths': [], 'neutral_relative_word_lengths': []}
      # if any words fail alignment consider it a corrupted file
      if all(word.case == "success" for word in alignment):
        for i, word in enumerate(alignment):
          new_row['base_word_lengths'].append(word.duration)
          # pauses are the difference between the end of one word and the start of the next
          if i+1 < len(alignment):
            new_row['pause_lengths'].append(alignment[i+1].start - word.end)
        # check if a neutral word length list exists for the given speaker and script
        neutral_lenghts = list(neutral_dataframe.loc[
          (neutral_dataframe['id'] == new_row['filename'].split('_')[0]) & 
          (neutral_dataframe['script'] == key) & 
          (neutral_dataframe['word_lengths'] != "NA"),
          'word_lengths'
        ])
        # determine word lengths relative to neutral as a percentage
        if len(neutral_lenghts):
          neutral_lenghts = neutral_lenghts[0]
          diff_ratio = (np.array(new_row["base_word_lengths"]) - np.array(neutral_lenghts))/(np.array(neutral_lenghts))
          new_row['neutral_relative_word_lengths'] = list(diff_ratio)
        else:
          new_row['neutral_relative_word_lengths'] = "NA"
      else:
        new_row['base_word_lengths'] = "NA"
        new_row['pause_lengths'] = "NA"
        new_row['neutral_relative_word_lengths'] = "NA"
      dataframe = pd.concat([dataframe, pd.Series(new_row).to_frame().T], ignore_index=True)

dataframe.to_csv('../data/word_length-base.csv', index = False) 