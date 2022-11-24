import pandas as pd
import statistics
import json

data = pd.read_csv("../data/clean_data.csv",converters={"base_word_lengths": lambda x: x.strip("[]").split(", "), "neutral_relative_word_lengths": lambda x: x.strip("[]").split(", "), "pause_lengths": lambda x: x.strip("[]").split(", ")})

# Sepreate neural data as it has no relative lengths
data_neutral = data[data["emotion"]  == "N"]
data_neutral.reset_index(drop=True, inplace=True)
data_neutral = data_neutral[["base_word_lengths", "script"]]

# non neutral
data = data[data["emotion"]  != "N"]
data.reset_index(drop=True, inplace=True)
data = data[["filename", "neutral_relative_word_lengths", "script", "emotion"]]

emotions = [
  "A",
  "D",
  "F",
  "H",
  "S"
]

scripts = [
  "It's eleven o'clock",
  "That is exactly what happened",
  "I'm on my way to the meeting",
  "I wonder what this is about",
  "The airplane is almost full",
  "Maybe tomorrow it will be cold",
  "I would like a new alarm clock",
  "I think I have a doctor's appointment",
  "Don't forget a jacket",
  "I think I've seen this before",
  "The surface is slick",
  "We'll stop in a couple of minutes"
]

# clean up rows with missing word lengths
for index, row in data_neutral.iterrows():
    for i in row["base_word_lengths"]:
        if i == '':
            data_neutral.drop(index, inplace=True)

# make list of lengths numbers
for index, row in data_neutral.iterrows():
    data_neutral["base_word_lengths"][index] = [float(i) for i in data_neutral["base_word_lengths"][index]]

script0 = data_neutral[data_neutral["script"] == scripts[0]]
script1 = data_neutral[data_neutral["script"] == scripts[1]]
script2 = data_neutral[data_neutral["script"] == scripts[2]]
script3 = data_neutral[data_neutral["script"] == scripts[3]]
script4 = data_neutral[data_neutral["script"] == scripts[4]]
script5 = data_neutral[data_neutral["script"] == scripts[5]]
script6 = data_neutral[data_neutral["script"] == scripts[6]]
script7 = data_neutral[data_neutral["script"] == scripts[7]]
script8 = data_neutral[data_neutral["script"] == scripts[8]]
script9 = data_neutral[data_neutral["script"] == scripts[9]]
script10 = data_neutral[data_neutral["script"] == scripts[10]]
script11 = data_neutral[data_neutral["script"] == scripts[11]]

script_dfs = [script0, script1, script2, script3, script4, script5, script6, script7, script8, script9, script10, script11]

average_word_lengths = {}

# get word lengths averaged over all factors for each script
for script_df in script_dfs:
  script = script_df['script'].iloc[0]
  average_word_lengths[script] = []
  for i in range(len(script_df['base_word_lengths'].iloc[0])):
    word_lengths = []
    for index, row in script_df.iterrows():
      word_lengths.append(row['base_word_lengths'][i])
    average_word_lengths[script].append(statistics.mean(word_lengths))

print("Writing")
print(average_word_lengths)
print("to /data/neutral_avg_word_length.json")
#save to JSON file
with open('../data/neutral_avg_word_length.json', 'w') as fp:
    json.dump(average_word_lengths, fp)

##########################################################################################
####################NON NEUTRAL DATA######################################################

# clean up rows with missing word lengths
for index, row in data.iterrows():
    for i in row["neutral_relative_word_lengths"]:
        if i == '':
            data.drop(index, inplace=True)

# make list of lengths numbers
for index, row in data.iterrows():
    data["neutral_relative_word_lengths"][index] = [float(i) for i in data["neutral_relative_word_lengths"][index]]

# add intensity and speaker columns
data['speaker'] = ""
data['intensity'] = ""

for index, row in data.iterrows():
  data["intensity"][index] = row["filename"].split("_")[-1]

for index, row in data.iterrows():
  data["speaker"][index] = row["filename"].split("_")[0]

average_word_lengths = {}

# get word lengths averaged over all factors for each script
for emotion in emotions:
  for script in scripts:
    average_word_lengths[script] = []
    df = data[(data["emotion"] == emotion) & (data["script"] == script)]
    for i in range(len(df['neutral_relative_word_lengths'].iloc[0])):
      word_lengths = []
      for index, row in df.iterrows():
        word_lengths.append(row['neutral_relative_word_lengths'][i])
      average_word_lengths[script].append(statistics.mean(word_lengths))
  with open(f'../data/{emotion}_avg_word_length.json', 'w') as fp:
    json.dump(average_word_lengths, fp)
  print("Writing")
  print(average_word_lengths)
  print(f"to /data/{emotion}_avg_word_length.json")


# if working by intensity
#for script_df in script_dfs:
#  script = script_df['script'].iloc[0]
#  average_word_lengths[script] = {"XX": [], "LO": [], "MD": [], "HI": []}
#  script_df = script_df.drop(["base_word_lengths"], axis = 1)
#  for column in script_df.loc[:, script_df.columns.str.contains('word')]:
#    avg_word_length = script_df.groupby(["intensity"])[column].mean()
#    average_word_lengths[script][avg_word_length.keys()[0]].append(avg_word_length[0])
#    average_word_lengths[script][avg_word_length.keys()[1]].append(avg_word_length[0])
#    average_word_lengths[script][avg_word_length.keys()[2]].append(avg_word_length[0])
