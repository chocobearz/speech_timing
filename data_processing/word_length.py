import os
import align

# implement pip as a subprocess:
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
#'pandas'])
import pandas as pd

'''
to run this code you will need to setup gentle aligner : https://github.com/lowerquality/gentle
use updated align script, and gentle folder included in this repo
to replace the one included in gentle aligner's original repo
'''


audioDirectory = r'/mnt/c/Users/ptut0/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/train/wav/'
alignerDirectory = r'/mnt/c/Users/ptut0/Documents/speech_timing/data_processing/gentle'
scriptDirectory = r'/mnt/c/Users/ptut0/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/train/txt/'
writeDirectory = '/mnt/c/Users/ptut0/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/train/'
data = pd.read_csv('/mnt/c/Users/ptut0/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/train/word_length-copy.csv')

for file in os.listdir(audioDirectory):
  filename = file.split('.')[0]
  if filename not in data['filename'].values:
    dataframe = pd.DataFrame({'filename' : [], 'word_lengths': [],'pause_lengths': []})
    script_location = scriptDirectory+filename+".txt" 
    with open(script_location, "r") as f:
      script = f.read()
    # only one key in filename guarenteeds
    print("aligning")
    alignment = align.align(audioDirectory+file, script)
    new_row = {'filename': filename, 'word_lengths': [], 'pause_lengths': []}
    for i, word in enumerate(alignment):
      if word.case == "success":
        new_row['word_lengths'].append(word.duration)
        # pauses are the difference between the end of one word and the start of the next
        if i+1 < len(alignment):
          if alignment[i+1].case == "success":
            new_row['pause_lengths'].append(round(alignment[i+1].start, 2) - round(word.end, 2))
          else:
            new_row['pause_lengths'].append("NA")
      else:
        new_row['word_lengths'].append("NA")
        new_row['pause_lengths'].append("NA")
    dataframe = pd.concat([dataframe, pd.Series(new_row).to_frame().T], ignore_index=True)
    dataframe.to_csv(writeDirectory+"word_length.csv", mode='a', header=not os.path.exists(writeDirectory+"word_length.csv"))
