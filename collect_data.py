import numpy as np
import torch

from transformers import BertTokenizer
from model import BertClassifier
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


device = 'cuda'
model_id  = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
model_dir = 'weights/bert_base_512.pth'

# Specify `MAX_LEN`
MAX_LEN = 512
batch_size=32

train_inputs = torch.load('data/loader/train_inputs.pt')
train_masks = torch.load('data/loader/train_masks.pt')
train_labels = torch.load('data/loader/train_labels.pt')

val_inputs = torch.load('data/loader/val_inputs.pt')
val_masks = torch.load('data/loader/val_masks.pt')
val_labels = torch.load('data/loader/val_labels.pt')

test_inputs = torch.load('data/loader/test_inputs.pt')
test_masks = torch.load('data/loader/test_masks.pt')
test_labels = torch.load('data/loader/test_labels.pt')

print(train_inputs.shape, train_labels.shape)
print(val_inputs.shape, val_labels.shape)
print(test_inputs.shape, test_labels.shape)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
BATCH_SIZE = 32

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


bert_classifier = BertClassifier()
bert_classifier.load_state_dict(torch.load(model_dir, map_location=device))
bert_classifier = bert_classifier.eval()
bert_classifier = torch.nn.DataParallel(bert_classifier)
bert_classifier = bert_classifier.to(device)


X_train = list()
y_train = list()

for batch in train_dataloader:
    b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        logits, outputs = bert_classifier(b_input_ids, b_attn_mask)
    
    xs = outputs[0][:, 0, :]
    preds = torch.argmax(logits, dim=1)
    
    for i in range(len(xs)):
        X_train.append(xs[i].tolist())
        y_train.append(preds[i].item())
                
X_train = np.array(X_train)
y_train = np.array(y_train)
np.save('data/WrapperData/X_train_whole.npy', X_train)
np.save('data/WrapperData/y_train_whole.npy', y_train)


X_val = list()
y_val = list()


for batch in val_dataloader:
    b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        logits, outputs = bert_classifier(b_input_ids, b_attn_mask)
    
    xs = outputs[0][:, 0, :]
    preds = torch.argmax(logits, dim=1)
    
    for i in range(len(xs)):
        X_val.append(xs[i].tolist())
        y_val.append(preds[i].item())
                
X_val = np.array(X_val)
y_val = np.array(y_val)
np.save('data/WrapperData/X_val_whole.npy', X_val)
np.save('data/WrapperData/y_val_whole.npy', y_val)


X_test = list()
y_test = list()


for batch in test_dataloader:
    b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        logits, outputs = bert_classifier(b_input_ids, b_attn_mask)
    
    xs = outputs[0][:, 0, :]
    preds = torch.argmax(logits, dim=1)
    
    for i in range(len(xs)):
        X_test.append( xs[i].tolist() )
        y_test.append(preds[i].item())
        
        
X_test = np.array(X_test)
y_test = np.array(y_test)
np.save('data/WrapperData/X_test_whole.npy', X_test)
np.save('data/WrapperData/y_test_whole.npy', y_test)










