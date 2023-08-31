import pandas as pd
import numpy as np
import os
import ast
import statistics

scriptDirectory = r"C:\\Users\\ptut0\\Documents\\speech_timing\\basline\\data\\txt\\clean\\"
writeDirectory = "C:\\Users\\ptut0\\Documents\\speech_timing\\basline\\data\\"
ssmlTags = ['x-weak', 'weak', 'medium-p', 'strong', 'x-strong']
pausePunct = [',', '!', '?', '.']
lenghtData = pd.read_csv('C:\\Users\\ptut0\\Documents\\speech_timing\\basline\\data\\word_length-copy.csv')

for index, row in lenghtData.iterrows():
  dataframe = pd.DataFrame({'filename' : [], 'discrete_pauses' : []})
  binLengths = {'filename': row['filename'], 'discrete_pauses' : []}
  # read string lists to real lists
  pauseLengths = ast.literal_eval(row['pause_lengths'])
  script_location = scriptDirectory+str(row['filename'])+".txt"
  with open(script_location, "r") as f:
    script = f.read()
  scriptList = script.split()
  # Only keep pauses at break punctuation
  naCount = pauseLengths.count("NA")
  if naCount < (len(pauseLengths)/4):
    if len(scriptList)-1 == len(pauseLengths):
      # interpolate NA values to mean
      lenMean = statistics.fmean(list(filter(lambda length: length != "NA", pauseLengths)))
      for i,pause in enumerate(pauseLengths):
        if pause == "NA":
          pauseLengths[i] = lenMean
      # get only non zero pentiles
      nonzPauseLengths = [i for i in pauseLengths if i > 0]
      pentiles = np.quantile(nonzPauseLengths, [0,0.2,0.4,0.6,0.8,1])
      # bin by pentiles
      pause_punct = []
      for i, pauseLength in enumerate(pauseLengths):
        if any(e in scriptList[i] for e in pausePunct):
          pause_punct.append(pauseLength)
          if pauseLength <= pentiles[1]:
            binLengths['discrete_pauses'].append(ssmlTags[0])
          elif pauseLength <= pentiles[2]:
            binLengths['discrete_pauses'].append(ssmlTags[1])
          elif pauseLength <= pentiles[3]:
            binLengths['discrete_pauses'].append(ssmlTags[2])
          elif pauseLength <= pentiles[4]:
            binLengths['discrete_pauses'].append(ssmlTags[3])
          elif pauseLength > pentiles[4]:
            binLengths['discrete_pauses'].append(ssmlTags[4])
          else:
            binLengths['discrete_pauses'] = ["ERROR-BIN"]
            dataframe = pd.concat([dataframe, pd.Series(binLengths).to_frame().T], ignore_index=True)
            dataframe.to_csv(writeDirectory+"discrete_pause_length.csv", mode='a', header=not os.path.exists(writeDirectory+"discrete_pause_length.csv"))
            continue
    else:
      binLengths['discrete_pauses'] = ["ERROR-MISMATCH"]
  else:
    binLengths['discrete_pauses'] = ["ERROR-NA"]
  dataframe = pd.concat([dataframe, pd.Series(binLengths).to_frame().T], ignore_index=True)
  dataframe.to_csv(writeDirectory+"discrete_pause_length.csv", mode='a', header=not os.path.exists(writeDirectory+"discrete_pause_length.csv"))
