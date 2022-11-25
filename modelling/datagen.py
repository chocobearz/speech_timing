import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

MAX_LEN = 0

def read_csv(filename, seed=0.9):
    data = pd.read_csv(filename, sep=',').fillna(0)
    # data = data.dropna(subset=['base_word_lengths'])
    data = data[data.emotion.isin(['A', 'N', 'H'])]
    return data


def get_pos_vector(postag_lst):
    unique_pos = {x for l in postag_lst for x in l}
    nunique_pos = len(unique_pos)

    pos_dict = dict()
    i = 0
    for p in unique_pos:
        x = [0]*(nunique_pos+1)
        x[i] = 1
        i += 1
        pos_dict[p] = x
    x = [0]*(nunique_pos+1)
    x[i] = 1
    pos_dict['pad'] = x

    pos_vectors = []
    for i,postags in enumerate(postag_lst):
        if len(postags) < MAX_LEN:
            postags.extend(['pad']*(MAX_LEN-len(postags)))
        pos_vectors.append([pos_dict[k] for k in postags])
    
    return pos_vectors


def get_word_embeddings(text_lst):
    unique_words = {x for l in text_lst for x in l}
    nunique_words = len(unique_words)
    print("Total number of unique words: ", nunique_words)

    word_dict = dict()
    i = 0
    for p in unique_words:
        x = [0]*(nunique_words+1)
        x[i] = 1
        i += 1
        word_dict[p] = x
    x = [0]*(nunique_words+1)
    x[i] = 1
    word_dict['pad'] = x

    text_vectors = []
    for i, wordtags in enumerate(text_lst):
        if len(wordtags) < MAX_LEN:
            wordtags.extend(['pad']*(MAX_LEN-len(wordtags)))
        text_vectors.append([word_dict[k] for k in wordtags])
    return text_vectors


def get_people_vector(person_lst):
    unique_people = set(person_lst)
    nunique_people = 91 #len(unique_people)
    print("Total number of unique people: ", nunique_people)

    word_dict = dict()
    i = 0
    for p in unique_people:
        x = [0]*(nunique_people)
        x[i] = 1
        i += 1
        word_dict[p] = x
    people_vectors = [word_dict[k] for k in person_lst]
    return people_vectors


def process_data(data):
    data = data.sort_values('emotion')
    text = [x.split() for x in data.script]
    text_vectors = get_word_embeddings(text)

    pos_seq = [x[1:-1].replace('\'','').replace(',','').split() for x in data.pos]
    pos_vectors = get_pos_vector(pos_seq)

    person_lst = [x.split('_')[0] for x in data.filename.tolist()]
    person_vec = get_people_vector(person_lst)

    # emotions vector
    emotions = data.emotion.tolist()
    # emotion_dict = {'A':0, 'D':1, 'F':2, 'H':3, 'N':4, 'S':5}
    emotion_dict = {'A':0, 'H':1, 'N':2}
    emotions_vec = []
    for i in range(len(emotions)):
        x = [0]*len(emotion_dict)
        x[emotion_dict[emotions[i]]] = 1
        emotions_vec.append(x)
    emotion_label = [emotion_dict[x] for x in emotions]

    # word lengths vector
    base_word_lengths = [len(x) for x in text]
    relative_word_length = data.neutral_relative_word_lengths.tolist()
    for i in range(len(emotions)):
        if relative_word_length[i] == 0:
            relative_word_length[i] = [0]*base_word_lengths[i]
        else:
            relative_word_length[i] = relative_word_length[i][1:-1].replace(',','').split()
            if len(relative_word_length[i]) < MAX_LEN:
                relative_word_length[i].extend([0]*(MAX_LEN-len(relative_word_length[i])))
            relative_word_length[i] = [float(x) if (float(x)<2.) else 2. for x in relative_word_length[i]]

    return np.array(relative_word_length), np.array(emotion_label), np.array(text_vectors), np.array(emotions_vec), np.array(person_vec), np.array(data.script.tolist())
    

class GetDataset(Dataset):
    def __init__(self, filename, val=False, seed=0.1):
        self.filename = filename
        dataframe = read_csv(self.filename, seed)
        self.relative_word_lengths, self.emotions, self.pos_vecs, self.emotions_vec, self.person_vec, self.script = process_data(dataframe)
        
        np.random.seed(10)
        indices = np.random.random_integers(0, high=len(self.emotions), size=int(0.9*len(self.emotions)))

        self.val = val
        # if val:
        #     self.relative_word_lengths = self.relative_word_lengths[~indices]
        # else:
        #     self.relative_word_lengths = self.relative_word_lengths[indices]

        print("Len of data {}".format(len(dataframe)))

    def __getitem__(self, idx):
        relative_word_length = torch.tensor(self.relative_word_lengths[idx], dtype=torch.float32)
        pos_vec = torch.tensor(self.pos_vecs[idx])
        emotions_vec  = torch.tensor(self.emotions_vec[idx])
        people_vec = torch.tensor(self.person_vec[idx])
        emotions_label = torch.tensor(self.emotions[idx])

        if self.val:
            return relative_word_length, emotions_label, emotions_vec, pos_vec, people_vec, self.script[idx]

        return relative_word_length, emotions_label, emotions_vec, pos_vec, people_vec
        
    def __len__(self):
        return len(self.emotions)


if __name__ == "__main__":

    filename = "data/clean_data.csv"
    GetDataset(filename)
    GetDataset(filename, val=True)


