import pandas as pd
import spacy

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")

processed_labelling = pd.read_csv('./CREMA-D-master/SentenceFilenames.csv')

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

emotion = {
    "ANG": "A",
    "DIS": "D",
    "FEA": "F",
    "HAP": "H",
    "NEU": "N",
    "SAD": "S"
}

for key in sentences:
    doc = nlp(sentences[key]['script'])
    for token in doc:
          sentences[key]['lemma'].append(token.lemma_),
          sentences[key]['pos'].append(token.pos_),
          sentences[key]['tag'].append(token.tag_),
          sentences[key]['dep'].append(token.dep_), 
          sentences[key]['stop'].append(token.is_stop), 
          sentences[key]['start'].append(token.is_sent_start), 
          sentences[key]['end'].append(token.is_sent_end)

labelling_data = processed_labelling[["Filename"]]

script = []
pos = []
dep = []
lemma = []
tag = []
stop = []
start = []
end = []

for file in labelling_data['Filename']:
  for key in sentences: 
    if key in file:
      script.append(sentences[key]["script"])
      pos.append(sentences[key]["pos"])
      dep.append(sentences[key]["dep"])
      lemma.append(sentences[key]["lemma"])
      tag.append(sentences[key]["tag"])
      stop.append(sentences[key]["stop"])
      start.append(sentences[key]["start"])
      end.append(sentences[key]["end"])
labelling_data["script"] = script
labelling_data["pos"] = pos
labelling_data["dep"] = dep
labelling_data["lemma"] = lemma
labelling_data["tag"] = tag
labelling_data["stop"] = stop
labelling_data["start"] = start
labelling_data["end"] = end

labelling_data["emotion"] = [emotion[key] for file in labelling_data['Filename'] for key in emotion if key in file]

labelling_data.to_csv('scriptData.csv', index = False) 