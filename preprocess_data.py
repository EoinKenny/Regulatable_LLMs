import torch
import pandas as pd

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

model_id = 'bert-base-uncased'
device = torch.device("cuda")



# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    for sent in data:
        # `encode_plus` will:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks




df = pd.read_csv('data/reviews.260k.train.txt', sep='\t', header=None)

for idx, row in df.iterrows():
    df.at[idx, 0] = float(row[0].split(" ")[4])


label = list()
for idx, row in df.iterrows():
    if row[0] > 0.7:
        label.append(1)
    else:
        label.append(0)

df['label'] = label
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df.to_csv('data/filtered_df.csv')
df = pd.read_csv('data/filtered_df.csv')
df = df.rename(columns={'1': 'STATEMENT', '0':'Rating'})

print(df.shape)
df_train1 = df[df.label==1]
df_train0 = df[df.label==0].sample( df_train1.shape[0], random_state=42 )
df = pd.concat([df_train0, df_train1])
print(df.shape)

del df['Unnamed: 0']
df.to_csv('data/final_df.csv')


# Number needed for validation
validation_size = int(df.shape[0] * 0.05)
print("Validation Size:", validation_size)

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame
# Splitting into training (90%), validation (5%), and testing (5%)
df_train, test_val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
df_val, df_test = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df["label"])

# Printing the sizes of each set
print("Training set size:", len(df_train))
print("Validation set size:", len(df_val))
print("Testing set size:", len(df_test))

df_train.to_csv('data/train_df.csv')
df_val.to_csv('data/val_df.csv')
df_test.to_csv('data/test_df.csv')

X_train = df_train.STATEMENT.values
y_train = df_train.label.values

X_val = df_val.STATEMENT.values
y_val = df_val.label.values

X_test = df_test.STATEMENT.values
y_test = df_test.label.values


# Specify `MAX_LEN`
MAX_LEN = 512
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks     = preprocessing_for_bert(X_val)
test_inputs, test_masks   = preprocessing_for_bert(X_test)

train_labels = torch.tensor(y_train, dtype=torch.long)
val_labels = torch.tensor(y_val, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

torch.save(train_inputs, 'data/loader/train_inputs.pt')
torch.save(train_masks,  'data/loader/train_masks.pt')
torch.save(train_labels, 'data/loader/train_labels.pt')

torch.save(val_inputs,   'data/loader/val_inputs.pt')
torch.save(val_masks,    'data/loader/val_masks.pt')
torch.save(val_labels,   'data/loader/val_labels.pt')

torch.save(test_inputs,  'data/loader/test_inputs.pt')
torch.save(test_masks,   'data/loader/test_masks.pt')
torch.save(test_labels,  'data/loader/test_labels.pt')









