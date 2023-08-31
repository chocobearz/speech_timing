import pandas as pd
import ast
import torch
import string
import numpy as np
from transformers import BertForTokenClassification
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizerFast
#disable parellelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

data = pd.read_csv("/localhome/ptuttosi/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/train/chunk_length.csv")
test_data = pd.read_csv("/localhome/ptuttosi/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/test/chunk_length.csv")
dev_data = pd.read_csv("/localhome/ptuttosi/Documents/speech_timing/TEDLIUM_release-3/speaker-adaptation/dev/chunk_length.csv")

#ssmlTags = set({'x-slow', 'slow', 'medium', 'fast', 'x-fast', 'x-weak', 'weak', 'medium-p', 'strong', 'x-strong'})
ssmlTags = set({'x-slow', 'slow', 'medium', 'fast', 'x-fast'})

# Map each label into its id representation and vice versa
#labels_to_ids = {'x-slow': 0, 'slow' : 1, 'medium' : 2, 'fast' : 3, 'x-fast': 4,  'x-weak' : 5, 'weak' : 6, 'medium-p' : 7, 'strong' : 8, 'x-strong' : 9}
#ids_to_labels = {0: 'x-slow', 1: 'slow', 2 : 'medium', 3 : 'fast', 4: 'x-fast', 5 : 'x-weak', 6 : 'weak', 7 : 'medium-p', 8 : 'strong', 9 : 'x-strong'}
labels_to_ids = {'x-slow': 0, 'slow' : 1, 'medium' : 2, 'fast' : 3, 'x-fast': 4}
ids_to_labels = {0: 'x-slow', 1: 'slow', 2 : 'medium', 3 : 'fast', 4: 'x-fast'}

def align_label(scripts, labels):
  # test removing all punctuation
  scripts.translate(str.maketrans('', '', string.punctuation)
  tokenized_inputs = tokenizer(scripts, padding='max_length', max_length=512, truncation=True)

  pausePunct = ',!?.'
  pauseTracker = 0

  word_ids = tokenized_inputs.word_ids()
  input_ids = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
  #print(f"labels: {len(labels)}")
  #print(f"pause labels: {len(pause_labels)}")
  #print(f"script: {len(scripts.split())}")
  #print(f" punct : {scripts.count('.') + scripts.count('!') + scripts.count('?') + scripts.count(',')}")

  previous_word_idx = None
  label_ids = []

  # to move forward at punctuation
  m = 0

  for i, word_idx in enumerate(word_ids):
    if word_idx is None:
      label_ids.append(-100)
    # at punctuation
    elif input_ids[i] in string.punctuation:
      label_ids.append(-100)
      m += 1
      # for a pause add a pause tag
      #if input_ids[i] in pausePunct:
      #  if word_ids[i + 1] != None:
      #    label_ids.append(labels_to_ids[pause_labels[pauseTracker]])
      #    pauseTracker += 1
      #    m += 1
      #  # end of sentence has no pause
      #  else:
      #    label_ids.append(-100)
      #    m += 1
      ## other punctuation no tag
      #else:
      #  label_ids.append(-100)
      #  m += 1
    # combine words with apostrophe to one word
    elif input_ids[i-1] == "'":
      label_ids.append(-100)
      m +=1
    # new word
    elif word_idx != previous_word_idx:
      try:
        label_ids.append(labels_to_ids[labels[word_idx-m]])
      except:
        label_ids.append(-100)
    # word with multiple tags
    else:
      label_ids.append(-100)
    previous_word_idx = word_idx
  return label_ids

class DataSequence(torch.utils.data.Dataset):

  def __init__(self, df):

    lb = [ast.literal_eval(i) for i in data['discrete_lengths'].values.tolist()]
    #plb = [ast.literal_eval(i) for i in df['discrete_pauses'].values.tolist()]
    txt = df['script'].values.tolist()
    #file = df['filename'].values.tolist()
    self.texts = [tokenizer(
        str(i),
        padding='max_length',
        max_length = 512,
        truncation=True,
        return_tensors="pt"
      ) for i in txt
    ]
    self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

  def __len__(self):
    return len(self.labels)

  def get_batch_data(self, idx):
    return self.texts[idx]

  def get_batch_labels(self, idx):
    return torch.LongTensor(self.labels[idx])

  def __getitem__(self, idx):
    batch_data = self.get_batch_data(idx)
    batch_labels = self.get_batch_labels(idx)
    return batch_data, batch_labels

class BertModel(torch.nn.Module):
  def __init__(self):
    super(BertModel, self).__init__()
    self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(ssmlTags))

  def forward(self, input_id, mask, label):
    output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
    return output

def train_loop(model, df_train, df_val):
  train_dataset = DataSequence(df_train)
  val_dataset = DataSequence(df_val)

  train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
  val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # can set momentum
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  if use_cuda:
    model = model.cuda()

  val_loss = []
  val_accuracy = []
  train_loss = []
  train_accuracy = []
  for epoch_num in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    model.train()

    for train_data, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_data['attention_mask'].squeeze(1).to(device)
      input_id = train_data['input_ids'].squeeze(1).to(device)
      optimizer.zero_grad()
      loss, logits = model(input_id, mask, train_label)

      for i in range(logits.shape[0]):
        # get rid of any with the hold label
        logits_clean = logits[i][train_label[i] != -100]
        label_clean = train_label[i][train_label[i] != -100]
        predictions = logits_clean.argmax(dim=1)
        acc = (predictions == label_clean).float().mean()
        total_acc_train += acc
        total_loss_train += loss.item()
      loss.backward()
      optimizer.step()
    #train_loss.append(total_loss_train/len(df_train))
    #train_accuracy.append(total_acc_train/len(df_train))
    model.eval()

    with torch.no_grad():
      total_acc_val = 0
      total_loss_val = 0

      for val_data, val_label in val_dataloader:
        # get rid of any with the hold label
        val_label = val_label.to(device)
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)
        loss, logits = model(input_id, mask, val_label)

        for i in range(logits.shape[0]):
          logits_clean = logits[i][val_label[i] != -100]
          label_clean = val_label[i][val_label[i] != -100]
          predictions = logits_clean.argmax(dim=1)
          acc = (predictions == label_clean).float().mean()
          total_acc_val += acc
          total_loss_val += loss.item()
    #val_loss.append(total_loss_val/len(df_train))
    #val_accuracy.append(total_acc_val/len(df_train))

    print(f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

def evaluate(model, df_test):

    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][test_label[i] != -100]
              label_clean = test_label[i][test_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')

def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)

# not done
def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


LEARNING_RATE = 0.0001
EPOCHS = 200
BATCH_SIZE = 16

model = BertModel()
train_loop(model, data, dev_data)
evaluate(model, test_data)
