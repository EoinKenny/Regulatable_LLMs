import torch
import pandas as pd
import pickle

from transformers import BertTokenizer
from model import BertClassifier, preprocessing_for_bert


model_id = 'bert-base-uncased'
data_type = '3Class'
DEVICE = "cpu"
MAX_LEN = 512
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)



def get_overlap(attr, current_idxs, df_row, colum_att_names):
    
    def overlap(range1, range2):
        if max(0, min(range1[1], range2[1]) - max(range1[0], range2[0])) > 0:
            # print(range1, range2)
            return 1
        else:
            return 0

    # Iterate over each column
    total = 0
    for col in colum_att_names:
        if col != attr:  # Avoid comparing a column with itself
            # print(col, attr)
            for other_idxs in df_row[col]:
                total += overlap(current_idxs, other_idxs)
    return total



bert_classifier = BertClassifier()
bert_classifier.load_state_dict(torch.load('weights/bert_base_512.pth', map_location=DEVICE))
bert_classifier = bert_classifier.eval()

columns = ['mouthfeel', 'aroma', 'appearance', 'taste', 'overall', 'raw', 'y', 'statement', 'rid']

df = pd.read_json('data/annotations.json', lines=True)

df.columns = columns



dfs = list()
colum_att_names = ['mouthfeel','aroma', 'appearance', 'taste', 'overall']

human_dfs = list()

for i, attr in enumerate(colum_att_names):
    df_temp = pd.DataFrame(columns=['statement', 'text', 'type', 'overlap'])
    statements, texts, types, overlaps = [], [], [], []
    
    for df_idx, df_row in df.iterrows():
        label_idxs = df_row[attr]
        
        for current_idxs in label_idxs:
            statements.append( ' '.join(df_row.statement) )
            texts.append( ' '.join(df_row.statement[current_idxs[0]:current_idxs[1]]) )
            types.append(attr)
            overlaps.append( get_overlap(attr, current_idxs, df_row, colum_att_names) )
            
    df_temp['statement'] = statements
    df_temp['text'] = texts
    df_temp['type'] = types
    df_temp['overlap'] = overlaps
    
    human_dfs.append( df_temp )



preds = list()

for i in range(len(human_dfs)):

    data = human_dfs[i]

    nb_tokens = [tokenizer.tokenize(sentence) for sentence in data.text.values.tolist()]
    input_ids, attention_masks = preprocessing_for_bert(data.text.values.tolist())   
    input_ids, attention_masks = input_ids.to(DEVICE), attention_masks.to(DEVICE)

    with torch.no_grad():
        logits, outputs = bert_classifier(input_ids, attention_masks)

    preds = torch.argmax(logits, dim=1).cpu().flatten().tolist()
    
    human_dfs[i]['pred'] = torch.argmax(logits, dim=1).cpu().flatten().tolist()


with open('data/human_dfs.pickle', 'wb') as f:
    pickle.dump(human_dfs, f)

with open('data/human_dfs.pickle', 'rb') as f:
    human_dfs = pickle.load(f)



train_human_dfs = list()
test_human_dfs = list()

for df in human_dfs:
    print(df.shape)
    
    # Split ratio, adjust as needed
    train_ratio = 0.8

    # Shuffle the DataFrame rows
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the number of rows for training and testing sets
    train_size = int(len(df_shuffled) * train_ratio)

    # Split the DataFrame into training and testing sets
    df_train = df_shuffled.iloc[:train_size]
    df_test = df_shuffled.iloc[train_size:]

    # Reset index for both DataFrames if needed
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)  
    
    train_human_dfs.append(df_train)
    test_human_dfs.append(df_test)
    print(df_train.shape, df_test.shape)


with open('data/train_human_dfs.csv', 'wb') as file:
    pickle.dump(train_human_dfs, file)

with open('data/test_human_dfs.csv', 'wb') as file:
    pickle.dump(test_human_dfs, file)








