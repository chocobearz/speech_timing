import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv("../data/word_length-base.csv",converters={"base_word_lengths": lambda x: x.strip("[]").split(", ")})

for index, row in data.iterrows():
    data["base_word_lengths"][index] = [float(i) for i in data["base_word_lengths"][index]]

sentences = {
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

emotions = {
    "angry": "A",
    "unfriendly": "D",
    "terrified": "F",
    "cheerful": "H",
    "sad": "S",
    "neutral": "N"
}

script = []
for index, row in data.iterrows():
  for key in sentences:  
    if key in row["filename"]:
      script.append(sentences[key])
data["script"] = script

emotion = []
for index, row in data.iterrows():
  for key in emotions:   
    if key in row["filename"]:
      emotion.append(emotions[key])
data["emotion"] = emotion

f = open("../data/A_avg_word_length_base.json")
anger = json.load(f)
f = open("../data/N_avg_word_length_base.json")
neutral = json.load(f)
f = open("../data/S_avg_word_length_base.json")
sad = json.load(f)
f = open("../data/H_avg_word_length_base.json")
happy = json.load(f)
f = open("../data/D_avg_word_length_base.json")
disgust = json.load(f)
f = open("../data/F_avg_word_length_base.json")
fear = json.load(f)

diff = []
rmse = []
pcc = []
rel_diff = []
for index, row in data.iterrows():
    if row['emotion'] == 'A':
      rmse.append(mean_squared_error(anger[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(anger[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(anger[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - anger[row["script"]][i]/word)
      rel_diff.append(rel_length)
    elif row['emotion'] == 'N':
      rmse.append(mean_squared_error(neutral[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(neutral[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(neutral[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - neutral[row["script"]][i]/word)
      rel_diff.append(rel_length)
    elif row['emotion'] == 'S':
      rmse.append(mean_squared_error(sad[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(sad[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(sad[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - sad[row["script"]][i]/word)
      rel_diff.append(rel_length)
    elif row['emotion'] == 'H':
      rmse.append(mean_squared_error(happy[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(happy[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(happy[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - happy[row["script"]][i]/word)
      rel_diff.append(rel_length)
    elif row['emotion'] == 'D':
      rmse.append(mean_squared_error(disgust[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(disgust[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(disgust[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - disgust[row["script"]][i]/word)
      rel_diff.append(rel_length)
    elif row['emotion'] == 'F':
      rmse.append(mean_squared_error(fear[row["script"]], row["base_word_lengths"], squared=False))
      pcc.append(np.corrcoef(fear[row["script"]], row["base_word_lengths"])[0][1])
      diff.append(list(np.subtract(fear[row["script"]], row["base_word_lengths"])))
      rel_length = []
      for i, word in enumerate(row["base_word_lengths"]):
        rel_length.append(word - fear[row["script"]][i]/word)
      rel_diff.append(rel_length)
data["diff"] = diff
data["rmse"] = rmse
data['pcc'] = pcc
data['relative_diff'] = rel_diff 

data = data[["filename", "base_word_lengths", "script", "emotion", "diff", "rmse", "pcc", "relative_diff"]]

data.to_csv("../data/objective_base_assesment.csv")


