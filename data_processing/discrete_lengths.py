import pandas as pd
import numpy as np
import os
import ast
import statistics

writeDirectory = "C:\\Users\\ptut0\\Documents\\speech_timing\\TEDLIUM_release-3\\speaker-adaptation\\test\\"
ssmlTags = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']
lenghtData = pd.read_csv('C:\\Users\\ptut0\\Documents\\speech_timing\TEDLIUM_release-3\\speaker-adaptation\\test\\chunk_length.csv')

def syllable_count(word):
  word = word.lower()
  count = 0
  vowels = "aeiouy"
  if word[0] in vowels:
    count += 1
  # a syllable is the number is distinct vowels
  for index in range(1, len(word)):
    if word[index] in vowels and word[index - 1] not in vowels:
      count += 1
  # silent e at the end of a word to not contribute to vowels
  if word.endswith("e"):
    count -= 1
  # silent e at the end of a word to not contribute to vowels
  if word.endswith("e's"):
    count -= 1
  # ending in ed does not contribute
  if word.endswith("ed"):
    count -= 1 
  # 'nt words have an extra syllable
  if "'nt" in word:
    count += 1
  if count == 0:
    count += 1
  return count

for index, row in lenghtData.iterrows():
  dataframe = pd.DataFrame({'filename' : [], 'discrete_lengths' : []})
  binLengths = {'filename': row['filename'], 'discrete_lengths' : []}
  # read string lists as real lists
  wordLengths = ast.literal_eval(row['word_lengths'])
  # if script is in the data file
  script = row['script']
  # uncomment below if running from txt scripts
  #with open(script_location, "r") as f:
  #  script = f.read()
  # need just the words as a list
  cleanScript = script.replace(",",'')
  cleanScript = cleanScript.replace(".",'')
  scriptList = cleanScript.split()
  # linear transformation relative to syllables
  # the lables are the pentiles (5 bins) specific to the speaker
  # interpolate NAs, if more than 1/4 of words are NA flag error
  naCount = wordLengths.count("NA")
  if naCount < (len(wordLengths)/4):
    # to simplify I changed hypens to spaces
    # to make robust to hypens need to adjust the pause and aligner code
    #if len(scriptList) != len(wordLengths):
    #  hyphens = [i for i, s in enumerate(scriptList) if "-" in s]
    #  if hyphens:
    #    for hyphen in hyphens:
    #      if wordLengths[hyphen+1] != "NA":
    #        if wordLengths[hyphen] != "NA":
    #          wordLengths[hyphen] = wordLengths[hyphen] + wordLengths[hyphen+1]
    #          del wordLengths[hyphen+1]
    #        else:
    #          del wordLengths[hyphen]
    #      else:
    #       del wordLengths[hyphen+1]
    if len(scriptList) != len(wordLengths):
      binLengths['discrete_lengths'] = ["ERROR-MISMATCH"]
      dataframe = pd.concat([dataframe, pd.Series(binLengths).to_frame().T], ignore_index=True)
      dataframe.to_csv(writeDirectory+"discrete_word_length-2.csv", mode='a', header=not os.path.exists(writeDirectory+"discrete_word_length-2.csv"))
      continue
    # interpolate NA values to mean
    lenMean = statistics.fmean(list(filter(lambda length: length != "NA", wordLengths)))
    for i,word in enumerate(wordLengths):
      if word == "NA":
        wordLengths[i] = lenMean
    # make all word lengths relative to sylls
    syllRelLength = [length / syllable_count(scriptList[word]) for word,length in enumerate(wordLengths)]

    pentiles = np.quantile(syllRelLength, [0,0.01,0.3,0.70,0.99,1])
    # bin by pentiles
    for wordLength in syllRelLength:
      if wordLength <= pentiles[1]:
        binLengths['discrete_lengths'].append(ssmlTags[4])
      elif wordLength <= pentiles[2]:
        binLengths['discrete_lengths'].append(ssmlTags[3])
      elif wordLength <= pentiles[3]:
        binLengths['discrete_lengths'].append(ssmlTags[2])
      elif wordLength <= pentiles[4]:
        binLengths['discrete_lengths'].append(ssmlTags[1])
      elif wordLength > pentiles[4]:
        binLengths['discrete_lengths'].append(ssmlTags[0])
      else:
        binLengths['discrete_lengths'] = ["ERROR-BIN"]
        dataframe = pd.concat([dataframe, pd.Series(binLengths).to_frame().T], ignore_index=True)
        dataframe.to_csv(writeDirectory+"discrete_word_length-2.csv", mode='a', header=not os.path.exists(writeDirectory+"discrete_word_length-2.csv"))
        continue
  else:
    binLengths['discrete_lengths'] = ["ERROR-NA"]
  dataframe = pd.concat([dataframe, pd.Series(binLengths).to_frame().T], ignore_index=True)
  dataframe.to_csv(writeDirectory+"discrete_word_length-2.csv", mode='a', header=not os.path.exists(writeDirectory+"discrete_word_length-2.csv"))
