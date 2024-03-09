import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import pickle
import time

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from fuzzywuzzy import fuzz
from model import BertClassifier, preprocessing_for_bert
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


model_id = 'bert-base-uncased'
data_type = '3Class'
DEVICE = "cuda"
MAX_LEN = 512
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
LATENT_SIZE = 768  # BERT
BATCH_SIZE = 1

# NB Set experiment type
EXPT_TYPES = ['avg', 'rnn', 'attention'] 
UNSUPERVISEDS = [False, True]
CONTEXT_AWARES = [False, True]



def replace_nans_0(tensor_with_nans):
    nan_mask = torch.isnan(tensor_with_nans)
    tensor_with_nans[nan_mask] = 0
    return tensor_with_nans



for EXPT_TYPE in EXPT_TYPES:
    for CONTEXT_AWARE in CONTEXT_AWARES:
        for UNSUPERVISED in UNSUPERVISEDS:

            start_time = time.time()



            bert_classifier = BertClassifier()
            bert_classifier.load_state_dict(torch.load('../../../weights/bert_base_512.pth', map_location=DEVICE))
            bert_classifier = bert_classifier.eval()
            bert_classifier = torch.nn.DataParallel(bert_classifier)
            bert_classifier = bert_classifier.to(DEVICE)           




            def split_statement_to_sentences(ids, token_ids=[1012, 1029, 999, 1010]):

                def split_paragraph(text):
                    pattern = r'[.?!]+|,'
                    sentences = re.split(pattern, text)
                    sentences = [sentence.strip() for sentence in sentences if sentence] 
                    return sentences    

                if (ids.shape[0] == 1):
                    ids = ids.flatten()

                text = tokenizer.decode(ids, skip_special_tokens=True)
                sentences = split_paragraph(text)

                return sentences


            def find_closest_partial_match_index(target_string, string_list, threshold=1):
                closest_match_index = None
                highest_similarity = 0
                for index, candidate_string in enumerate(string_list):
                    similarity = fuzz.partial_ratio(target_string, candidate_string)
                    if similarity > highest_similarity and similarity >= threshold:
                        highest_similarity = similarity
                        closest_match_index = index
                return closest_match_index    


            def return_match_label(original_statement, target_sequence, split_statement):
                labels = [0 for _ in range(len(split_statement))]
                matching_idx = find_closest_partial_match_index(target_sequence, split_statement)
                labels[matching_idx] = 1
                return labels  


            def get_max_2d_idx(tensor):
                # Get the index of the highest value
                index = torch.argmax(tensor)
                row_index = index // tensor.shape[1]
                col_index = index % tensor.shape[1]

                return row_index.item(), col_index.item()


            def new_func(input_ids_statement, input_ids_label, attention_masks_label, outputs, encoder):
                """
                sample the human labelled concept data with context
                """

                def find_sequence_indices(sequence, target_tensor):
                    indices = []
                    sequence_length = len(sequence)
                    for i in range(len(target_tensor) - sequence_length + 1):
                        if torch.equal(target_tensor[i:i + sequence_length], sequence):
                            indices.append([i, i + sequence_length])
                    return indices[0]

                input_ids_statement = input_ids_statement[0][1:-1]
                annotation_indicies = find_sequence_indices(  input_ids_label[attention_masks_label==1][1:-1], input_ids_statement  )

                # print(annotation_indicies)

                if encoder != None:
                    emb_seq = outputs[0][0][ annotation_indicies[0]: annotation_indicies[1], : ]
                    emb_seq = encoder(emb_seq.unsqueeze(0))
                else:
                    # print(outputs[0].shape)

                    emb_seq = outputs[0][0][ annotation_indicies[0]: annotation_indicies[1], : ].mean(dim=0).unsqueeze(0)

                return emb_seq            
            
            
            
            
            
            
            
            
            def eval_concepts(dfs_data):


                data = list()

                if not UNSUPERVISED:
                    prototype_embeddings_mean, _ = sample_labels(dfs_train, 4, bert_classifier, encoder, sample_full=True)
                else:
                    prototype_embeddings_mean = None


                predictions = list()
                labels = list()
                activations = list()

                # Iterated all examples of one class
                for i in range(len(dfs_data)):

                    for j, split in enumerate(['yes', 'no']):

                        datat = list()

                        temp_df = dfs_data[i]
                        temp_df = temp_df[temp_df.label==split]

                        for k in range(len(temp_df)):

                            input_ids_label, attention_masks_label         = preprocessing_for_bert([temp_df.iloc[k].text])
                            input_ids_statement, attention_masks_statement = preprocessing_for_bert([temp_df.iloc[k].statement])


                            if not CONTEXT_AWARE:
                                with torch.no_grad():
                                    _, outputs = bert_classifier(input_ids_label, attention_masks_label)
                                    emb = outputs[0][:, 0, :]

                            else:
                                with torch.no_grad():
                                    _, outputs = bert_classifier(input_ids_statement, attention_masks_statement)
                                    emb = new_func(input_ids_statement, input_ids_label, attention_masks_label, outputs, encoder)

                            _, acts = pw_net(emb.to(DEVICE), prototype_embeddings_mean)

                            highest_each_row = torch.max(acts, dim=1)[0]
                            max_acts, max_idxs = torch.topk(highest_each_row, k=3)
                            pred = max_idxs.tolist()


                            predictions.append(pred)
                            labels.append( ((i*2)+j) )
                            activations.append(acts.tolist())


                predictions = np.array(predictions)
                top1_preds = predictions.T[0]

                top3_preds = list()
                for i in range(len(labels)):
                    if labels[i] in predictions[i]:
                        top3_preds.append(labels[i])
                    else:
                        top3_preds.append(top1_preds[i])

                return accuracy_score(top1_preds, labels)            
            
            
            
            


            def replace_nans_0(tensor_with_nans):
                nan_mask = torch.isnan(tensor_with_nans)
                tensor_with_nans[nan_mask] = 0
                return tensor_with_nans            

            
            
            
            

            with open('../data/train_human_dfs.csv', 'rb') as file:
                dfs_train = pickle.load(file)





            def remove_bad_labels(bert_classifier, dfs):
                preds = list()
                keys = [ [[1,2], [0]],
                         [[0], [1,2]],
                         [[0,1], [2]],
                         [[2], [0,1]]
                       ]
                for i in range(len(dfs)):
                    df = dfs[i]
                    temp = df[((df['label'] == 'yes') & (df['bb_anno_pred'].isin(keys[i][0]))) | ((df['label'] == 'no') & (df['bb_anno_pred'].isin(keys[i][1]) ))]
                    dfs[i] = temp
                return dfs

            dfs_train = remove_bad_labels(bert_classifier, dfs_train)







            def sample_labels_contextAware(dfs, sample_size, bert_classifier, encoder, sample_full=False, take_mean=True):
                """
                sample the human labelled concept data with context
                """

                def find_sequence_indices(sequence, target_tensor):
                    indices = []
                    sequence_length = len(sequence)
                    for i in range(len(target_tensor) - sequence_length + 1):
                        if torch.equal(target_tensor[i:i + sequence_length], sequence):
                            indices.append([i, i + sequence_length])
                    return indices

                embeddings = list()
                preds = list()

                for i in range(len(dfs)):

                    if sample_full:
                        yes_df   = dfs[i][(dfs[i].label=='yes') & (pd.notna(dfs[i].text))]
                        no_df = dfs[i][(dfs[i].label=='no')  & (pd.notna(dfs[i].text))]
                    else:
                        yes_df   = dfs[i][(dfs[i].label=='yes') & (pd.notna(dfs[i].text))].sample(sample_size)
                        no_df = dfs[i][(dfs[i].label=='no')  & (pd.notna(dfs[i].text))].sample(sample_size)


                    for temp_df in [yes_df, no_df]:
                        statements  = temp_df.statement.values.tolist()
                        annotations = temp_df.text.values.tolist()

                        statement_tokens  = [tokenizer.tokenize(statement) for statement in statements]
                        annotation_tokens = [tokenizer.tokenize(annotation) for annotation in annotations]

                        input_ids, attention_masks = preprocessing_for_bert(statement_tokens)   
                        label_seq, label_masks = preprocessing_for_bert(annotation_tokens) 

                        annotation_indicies = [ find_sequence_indices(label_seq[j][label_masks[j]==1][1:-1], input_ids[j])[0] for j in range(len(label_seq)) ]

                        input_ids, attention_masks = input_ids.to(DEVICE), attention_masks.to(DEVICE)


                        with torch.no_grad():
                            logits, outputs = bert_classifier(input_ids, attention_masks)


                        # print(outputs[0].shape)


                        # Maximum len of sequences
                        padding_len = max([x[1] - x[0] for x in annotation_indicies])

                        if encoder != None:
                            # print("Encoder Found")
                            encoder_inputs = torch.zeros(sample_size, padding_len, LATENT_SIZE).to(DEVICE)
                            for j in range(sample_size):
                                emb_seq = outputs[0][j][ annotation_indicies[j][0]: annotation_indicies[j][1] ]
                                # print(emb_seq)
                                encoder_inputs[j][:emb_seq.shape[0], :] = emb_seq
                            final_embeddings = encoder(encoder_inputs)
                        else:
                            # print("No Encoder found")
                            final_embeddings = list()
                            for j in range(sample_size):
                                emb_seq = outputs[0][j][ annotation_indicies[j][0]: annotation_indicies[j][1] ]
                                # print("appending this:", emb_seq.mean(dim=0).shape)
                                final_embeddings.append(emb_seq.mean(dim=0))                     
                            final_embeddings = torch.stack(final_embeddings, axis=0)

                        embeddings.append( final_embeddings.mean(axis=0) )

                        preds.append( torch.argmax(logits, dim=1).cpu().flatten().tolist() )

                embeddings = torch.stack(embeddings, axis=0)

                # print("final embeds of human data samples:", embeddings.shape)

                return embeddings.to(DEVICE), preds



            def sample_labels_contextUnaware(dfs, sample_size, bert_classifier, sample_full=False, take_mean=True):
                """
                Take a bunch of human labelled data
                pass into bert
                avg the cls tokens
                """

                embeddings = list()  # should return shape == (8, sample_size)

                preds = list()

                for i in range(len(dfs)):

                    if sample_full:
                        data1 = dfs[i][(dfs[i].label=='yes') & (pd.notna(dfs[i].text))]
                        data2 = dfs[i][(dfs[i].label=='no') & (pd.notna(dfs[i].text))]
                    else:
                        data1 = dfs[i][(dfs[i].label=='yes') & (pd.notna(dfs[i].text))].sample(sample_size)
                        data2 = dfs[i][(dfs[i].label=='no') & (pd.notna(dfs[i].text))].sample(sample_size)

                    for data in [data1, data2]:
                        input_ids, attention_masks = preprocessing_for_bert(data.text.values.tolist())   
                        input_ids, attention_masks = input_ids.to(DEVICE), attention_masks.to(DEVICE)

                        with torch.no_grad():
                            logits, outputs = bert_classifier(input_ids, attention_masks)

                        token_embeddings = outputs[0][:, 0, :]   

                        if take_mean:
                            embeddings.append(token_embeddings.mean(dim=0))
                            preds.append( torch.argmax(logits, dim=1).cpu().flatten().tolist() )
                        else:
                            embeddings.append(token_embeddings)
                            preds.append( torch.argmax(logits, dim=1).cpu().flatten().tolist() )

                return torch.stack(embeddings), preds







            def sample_labels(dfs, sample_size, bert_classifier, rnn_encoder, sample_full=False, take_mean=True):

                if CONTEXT_AWARE:
                    return sample_labels_contextAware(dfs, sample_size, bert_classifier, rnn_encoder, sample_full=sample_full, take_mean=take_mean)
                else:
                    return sample_labels_contextUnaware(dfs, sample_size, bert_classifier, sample_full=sample_full, take_mean=take_mean)






            df_train = pd.read_csv('../../../data/'+data_type+'/train_df.csv')
            df_val   = pd.read_csv('../../../data/'+data_type+'/val_df.csv')
            df_test  = pd.read_csv('../../../data/'+data_type+'/test_df.csv')


            #### Load dataset loaders
            train_inputs = torch.load('../../../data/'+data_type+'/loader/train_inputs.pt')
            train_masks = torch.load('../../../data/'+data_type+'/loader/train_masks.pt')
            val_inputs = torch.load('../../../data/'+data_type+'/loader/val_inputs.pt')
            val_masks = torch.load('../../../data/'+data_type+'/loader/val_masks.pt')
            test_inputs = torch.load('../../../data/'+data_type+'/loader/test_inputs.pt')
            test_masks = torch.load('../../../data/'+data_type+'/loader/test_masks.pt')

            train_labels = torch.load('../../../data/'+data_type+'/loader/train_labels.pt')
            val_labels = torch.load('../../../data/'+data_type+'/loader/val_labels.pt')
            test_labels = torch.load('../../../data/'+data_type+'/loader/test_labels.pt')

            df_train = pd.read_csv('../../../data/'+data_type+'/train_df.csv')
            df_val = pd.read_csv('../../../data/'+data_type+'/val_df.csv')
            df_test = pd.read_csv('../../../data/'+data_type+'/test_df.csv')



            # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
            batch_size = 1

            # Create the DataLoader for our training set
            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

            # Create the DataLoader for our validation set
            val_data = TensorDataset(val_inputs, val_masks, val_labels)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

            # Create the DataLoader for our test set
            test_data = TensorDataset(test_inputs, test_masks, test_labels)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

            def get_sent_embeddings_from_batch_contextUnaware(ids, token_ids=[1012, 1029, 999, 1010]):
                """
                Preprocess statement into sentences
                return: latent representation of text sentences without any context of text around them
                """

                def split_paragraph(text):
                    pattern = r'[.?!]+|,'
                    sentences = re.split(pattern, text)
                    sentences = [sentence.strip() for sentence in sentences if sentence] 
                    return sentences    

                if (ids.shape[0] == 1):
                    ids = ids.flatten()

                text = tokenizer.decode(ids, skip_special_tokens=True)
                sentences = split_paragraph(text)
                input_ids, input_masks = preprocessing_for_bert(sentences)

                with torch.no_grad():
                    _, outputs = bert_classifier(input_ids, input_masks)

                return outputs[0][:, 0, :]




            def get_sent_embeddings_from_batch_contextAware(b_input_ids, b_attn_mask, outputs, encoder):

                outputs = outputs[0][:, 1:, :][0] # remove [CLS]
                b_input_ids = b_input_ids[:, 1:]  # remove [CLS]
                b_attn_mask = b_attn_mask[:, 1:]  # remove [CLS]


                # batch size = num sentences
                # sequence len = max sent len
                # latent dim...
                numbers_to_find = [1012, 1029, 999, 1010]
                indexes = torch.where(torch.isin(b_input_ids, torch.tensor(numbers_to_find).to(DEVICE)))[1]

                # add beginning and end to indexes
                beginning_tensor = torch.tensor([0]).to(DEVICE)
                end_tensor = torch.tensor([b_attn_mask.sum().item() - 2]).to(DEVICE)
                indexes = torch.cat([beginning_tensor, indexes, end_tensor])
                seq_len, _ = torch.max(torch.diff(indexes), dim=0)

                # for when the last two indexes are the same
                if indexes[-2] == indexes[-1]:
                    indexes = indexes[:-1]

                # if there is an encoder
                if encoder != None:
                    wrapper_input = torch.zeros(indexes.shape[0]-1, seq_len, LATENT_SIZE).to(DEVICE)
                    for i in range(len(indexes)-1):
                        sentence_emb = outputs[ indexes[i].item(): indexes[i+1].item(), : ]
                        wrapper_input[ i, :sentence_emb.shape[0], : ] = sentence_emb
                    encoder_embeddings = encoder(wrapper_input)
                else:
                    encoder_embeddings = list()
                    for i in range(len(indexes)-1):
                        sentence_emb = outputs[ indexes[i].item(): indexes[i+1].item(), : ]
                        encoder_embeddings.append(sentence_emb.mean(dim=0))
                    encoder_embeddings = torch.stack(encoder_embeddings, axis=0)

                return encoder_embeddings

            def get_sent_embeddings_from_batch(b_input_ids, b_attn_mask, outputs, encoder):

                if CONTEXT_AWARE:
                    return get_sent_embeddings_from_batch_contextAware(b_input_ids, b_attn_mask, outputs, encoder)
                else:
                    return get_sent_embeddings_from_batch_contextUnaware(b_input_ids)


            MODEL_DIR = '../../../weights/pwnet-multi.pth'
            ENC_DIR = '../../../weights/enc.pth'
            NUM_CLASSES = 3
            PROTOTYPE_SIZE = 8
            NUM_EPOCHS = 2
            NUM_PROTOTYPES = 8




            class ListModule(object):
                #Should work with all kind of module
                def __init__(self, module, prefix, *args):
                    self.module = module
                    self.prefix = prefix
                    self.num_module = 0
                    for new_module in args:
                        self.append(new_module)

                def append(self, new_module):
                    if not isinstance(new_module, nn.Module):
                        raise ValueError('Not a Module')
                    else:
                        self.module.add_module(self.prefix + str(self.num_module), new_module)
                        self.num_module += 1

                def __len__(self):
                    return self.num_module

                def __getitem__(self, i):
                    if i < 0 or i >= self.num_module:
                        raise IndexError('Out of bound')
                    return getattr(self.module, self.prefix + str(i))




            import torch.nn.functional as F
            class AttentionLayer(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super(AttentionLayer, self).__init__()
                    self.W_query = nn.Linear(input_size, hidden_size)
                    self.W_key = nn.Linear(input_size, hidden_size)
                    self.v = nn.Linear(hidden_size, 1)
                    self.dropout = nn.Dropout(0.1)

                def forward(self, encoder_outputs):
                    query = self.W_query(encoder_outputs)
                    key = self.W_key(encoder_outputs)

                    energy = torch.tanh(query + key)
                    attention_scores = F.softmax(self.v(energy), dim=1)
                    attention_scores = self.dropout(attention_scores)

                    context = torch.sum(encoder_outputs * attention_scores, dim=1)

                    return context





            class AttentionEncoder(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super(AttentionEncoder, self).__init__()
                    self.attention = AttentionLayer(input_size, hidden_size)

                def forward(self, x):
                    final_embedding = self.attention(x)
                    return final_embedding






            class RNNEncoder(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True):
                    super(RNNEncoder, self).__init__()
                    self.rnn = nn.LSTM(input_size,
                                       hidden_size,
                                       num_layers,
                                       bidirectional=bidirectional,
                                       batch_first=True,
                                       dropout=0.5
                                      )

                def forward(self, x):
                    _, (h_n, _) = self.rnn(x)
                    # Assuming you want the last hidden state of the last layer
                    return h_n[-1]






            class PWNet(nn.Module):

                def __init__(self):
                    super(PWNet, self).__init__()
                    self.ts = ListModule(self, 'ts_')
                    for i in range(NUM_PROTOTYPES):
                        transformation = nn.Sequential(
                            nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                            nn.InstanceNorm1d(PROTOTYPE_SIZE),
                            nn.ReLU(),
                            nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
                        )
                        self.ts.append(transformation)  

                    if UNSUPERVISED:
                        prototypes = torch.randn( (NUM_PROTOTYPES, LATENT_SIZE), dtype=torch.float32 )
                        self.prototypes = nn.Parameter(prototypes, requires_grad=True)
                    else:
                        prototypes = torch.randn( (NUM_PROTOTYPES, LATENT_SIZE), dtype=torch.float32 )
                        self.prototypes = nn.Parameter(prototypes, requires_grad=False)

                    self.epsilon = 1e-4
                    self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=True) 

                    if not UNSUPERVISED:
                        self.__make_linear_weights()

                def __make_linear_weights(self):

                    custom_weight_matrix = torch.tensor([
                                                         [ -1.,  1.,  1. ], # IV Liable
                                                         [  1., -1., -1. ], # IV Not liable
                                                         [  1., -1., -1. ], # IV Def
                                                         [ -1.,  1.,  1. ], # IV Not Def
                                                         [  1.,  1., -1. ], 
                                                         [ -1., -1.,  1. ],
                                                         [ -1., -1.,  1. ], 
                                                         [  1.,  1., -1. ],
                    ])

                    self.linear.weight.data.copy_(custom_weight_matrix.T)       


                def __proto_layer_l2(self, x, p):
                    output = list()
                    n_sent = x.shape[0]
                    n_protos = p.shape[0]

                    x = x.unsqueeze(1).repeat(1, n_protos, 1).to(DEVICE)
                    p = p.unsqueeze(0).repeat(n_sent, 1, 1).to(DEVICE)
                    l2s = torch.norm(x - p, dim=2, p=2).pow(2).to(DEVICE)        
                    all_acts = torch.log((l2s + 1.) / (l2s + self.epsilon)).to(DEVICE)
                    max_act, idx = all_acts.max(dim=0)

                    # print(all_acts)
                    # print(max_act)
                    # print(idx)
                    # print(" ")

                    return all_acts, max_act, idx

                def forward(self, x, prototypes=None):

                    # If using human prototypes
                    if not UNSUPERVISED:
                        self.prototypes.data.copy_(prototypes)

                    all_activations = list()
                    max_activations = list()
                    idxs = list()

                    for i, t in enumerate(self.ts):

                        prototype = t( self.prototypes[i].view(1,-1) )
                        sent_trans = t(x)

                        all_acts, max_act, idx = self.__proto_layer_l2( sent_trans, prototype )

                        all_acts = all_acts.view(1,-1)
                        max_act = max_act.view(1, -1)

                        all_activations.append( all_acts )
                        max_activations.append( max_act )
                        idxs.append( idx.item() )

                    all_activations = torch.cat(all_activations, axis=0)
                    max_activations = torch.cat(max_activations, axis=0)

                    logits = self.linear(max_activations.T)

                    return logits, all_activations



            def evaluate_loader(wrapper, encoder, black_box, loader, loss):

                wrapper.eval()

                if encoder != None:
                    encoder.eval()
                total_error = 0
                total = 0
                correct = 0

                if not UNSUPERVISED:
                    prototype_embeddings_mean, _ = sample_labels(dfs_train, 4, bert_classifier, encoder, sample_full=True)
                else:
                    prototype_embeddings_mean = None

                with torch.no_grad():
                    for i, data in enumerate(loader):

                        # Prepare data (don't need labels)
                        b_input_ids, b_attn_mask, actual_labels = data
                        b_input_ids, b_attn_mask, actual_labels = b_input_ids.to(DEVICE), b_attn_mask.to(DEVICE), actual_labels.to(DEVICE)

                        logits, outputs = black_box(b_input_ids, b_attn_mask)
                        bb_labels = torch.argmax(logits, dim=1)

                        try:
                            sent_input_ids = get_sent_embeddings_from_batch(b_input_ids, b_attn_mask, outputs, encoder)

                            # Check for NaN values
                            if prototype_embeddings_mean != None:
                                if True in torch.isnan(prototype_embeddings_mean):
                                    print('found NaN in prototype_embeddings_mean')
                                    prototype_embeddings_mean = replace_nans_0(prototype_embeddings_mean)
                            if True in torch.isnan(sent_input_ids):
                                print('found NaN in sent_input_ids')
                                sent_input_ids = replace_nans_0(sent_input_ids)

                            logits, _ = pw_net(sent_input_ids.to(DEVICE), prototype_embeddings_mean)

                            if True in torch.isnan(logits):
                                print('found NaN in logits')
                                logits = replace_nans_0(logits)

                        except:
                            print("Failed forward pass")
                            continue                        

                        current_loss = loss(logits, actual_labels)
                        total_error += current_loss.item()
                        total += len(b_input_ids)
                        correct += (  torch.argmax(logits, dim=1) == actual_labels  ).sum()            

                wrapper.train()
                if encoder != None:
                    encoder.train()

                return round(total_error / total, 6), round(correct.item() / total, 4)


            with open('../data/test_human_dfs.csv', 'rb') as file:
                dfs_test = pickle.load(file)  



            print("\n\n =============================================")
            print("Experiment Type:", EXPT_TYPE)
            print("Unsupervised:", UNSUPERVISED)
            print("Context Aware:", CONTEXT_AWARE)


            pw_net = PWNet()
            pw_net = pw_net.to(DEVICE).train()

            if EXPT_TYPE == 'rnn':
                encoder = RNNEncoder(LATENT_SIZE, LATENT_SIZE, num_layers=2, bidirectional=True)
                encoder.to(DEVICE)
                encoder.train()
            elif EXPT_TYPE == 'attention':
                encoder = AttentionEncoder(LATENT_SIZE, LATENT_SIZE)
                encoder.to(DEVICE)
                encoder.train()
            else:
                encoder = None

            cce_loss = nn.CrossEntropyLoss()

            # Add parameters for training optimizer
            trained_parameters = list(pw_net.parameters())
            if encoder != None:
                trained_parameters += list(encoder.parameters())

            optimizer = torch.optim.Adam( trained_parameters, lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            best_acc = 0.
            running_loss = 0
            training_sample_num = 8

            concept_loss_data = list()
            normal_loss_data = list()


            for epoch in range(NUM_EPOCHS):

                epoch_start_time = time.time()
                fail_forward_pass = 0

                for i, data in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    # Prepare data (don't need labels)
                    b_input_ids, b_attn_mask, labels = data
                    b_input_ids, b_attn_mask, labels = b_input_ids.to(DEVICE), b_attn_mask.to(DEVICE), labels.to(DEVICE)

                    with torch.no_grad():
                        bb_logits, outputs = bert_classifier(b_input_ids, b_attn_mask)
                        bb_labels = torch.argmax(bb_logits, dim=1)


                    try:
                        # Run inference on wrapper
                        if not UNSUPERVISED: prototype_embeddings_mean, _ = sample_labels(dfs_train, training_sample_num, bert_classifier, encoder, sample_full=False)
                        else: prototype_embeddings_mean = None
                        sent_input_ids = get_sent_embeddings_from_batch(b_input_ids, b_attn_mask, outputs, encoder)

                        # Check for NaN values
                        if prototype_embeddings_mean != None:
                            if True in torch.isnan(prototype_embeddings_mean):
                                print('found NaN in prototype_embeddings_mean')
                                prototype_embeddings_mean = replace_nans_0(prototype_embeddings_mean)
                        if True in torch.isnan(sent_input_ids):
                            print('found NaN in sent_input_ids')
                            sent_input_ids = replace_nans_0(sent_input_ids)

                        logits, _ = pw_net(sent_input_ids.to(DEVICE), prototype_embeddings_mean)

                        if True in torch.isnan(logits):
                            print('found NaN in logits')
                            logits = replace_nans_0(logits)

                    except:
                        print("Failed forward pass")
                        fail_forward_pass += 1
                        continue


                    loss1 = cce_loss(logits, labels.detach().to(DEVICE))


                    # take concept loss
                    prototype_embeddings, _ = sample_labels(dfs_train, training_sample_num, bert_classifier, encoder, sample_full=False, take_mean=False)
                    concept_train_preds  = torch.zeros((64,8), dtype=torch.float32)
                    for p_idx, p in enumerate(prototype_embeddings.view(-1, 768)):
                        concept_train_preds[p_idx] = pw_net(p.view(1,-1), prototype_embeddings_mean)[1].flatten()
                    concept_train_lables = torch.cat([torch.tensor([i]*8, dtype=torch.long) for i in range(8)])   
                    loss2 = cce_loss(concept_train_preds, concept_train_lables)

                    loss = loss1 + loss2

                    concept_loss_data.append(loss2.item())
                    normal_loss_data.append(loss1.item())

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()


                    if i % 200 == 0:
                        print("time taken:", time.time() - epoch_start_time)

                        _, current_label_acc = evaluate_loader(pw_net, encoder, bert_classifier, val_dataloader, cce_loss)

                        current_acc = eval_concepts(dfs_test)

                        print("Current Label Acc:", current_label_acc)
                        print("Current Concept Acc:", current_acc)
                        print('Loss:', running_loss / (i+1))
                        print("Batch:", i)
                        running_loss = 0

                        if current_acc > best_acc:
                            torch.save(pw_net.state_dict(), MODEL_DIR)
                            if encoder != None:
                                torch.save(encoder.state_dict(), ENC_DIR)

                        best_acc = current_acc
                        print('saving model')
                        print(" ")
                        scheduler.step()


                    if UNSUPERVISED:
                        if i == 5000:
                            break
                    else:
                        if i == 15000:
                            break



            # Plotting
            plt.plot(concept_loss_data, label='Concept-Loss', alpha=0.5)
            plt.plot(normal_loss_data, label='Normal-Loss', alpha=0.5)

            # Adding labels and legend
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Concept Loss vs. Normal Loss')
            plt.legend()

            # Display the plot
            plt.savefig('plots/' + EXPT_TYPE + str(UNSUPERVISED) + str(CONTEXT_AWARE) + 'losses.pdf')            





            if EXPT_TYPE == 'rnn':
                encoder = RNNEncoder(LATENT_SIZE, LATENT_SIZE, num_layers=2, bidirectional=True)
                encoder.load_state_dict(torch.load(ENC_DIR))
                encoder.to(DEVICE)
                encoder.eval()
            elif EXPT_TYPE == 'attention':
                encoder = AttentionEncoder(LATENT_SIZE, LATENT_SIZE)
                encoder.load_state_dict(torch.load(ENC_DIR))
                encoder.to(DEVICE)
                encoder.eval()
            else:
                encoder = None

            pw_net = PWNet()
            pw_net.load_state_dict(torch.load(MODEL_DIR, map_location='cpu'))
            pw_net.eval()
            pw_net = pw_net.to(DEVICE)

            _, test_acc_data = evaluate_loader(pw_net, encoder, bert_classifier, test_dataloader, cce_loss)

            with open('../data/test_human_dfs.csv', 'rb') as file:
                dfs_test = pickle.load(file)  


            def split_statement_to_sentences(ids, token_ids=[1012, 1029, 999, 1010]):

                def split_paragraph(text):
                    pattern = r'[.?!]+|,'
                    sentences = re.split(pattern, text)
                    sentences = [sentence.strip() for sentence in sentences if sentence] 
                    return sentences    

                if (ids.shape[0] == 1):
                    ids = ids.flatten()

                text = tokenizer.decode(ids, skip_special_tokens=True)
                sentences = split_paragraph(text)

                return sentences


            def find_closest_partial_match_index(target_string, string_list, threshold=1):
                closest_match_index = None
                highest_similarity = 0
                for index, candidate_string in enumerate(string_list):
                    similarity = fuzz.partial_ratio(target_string, candidate_string)
                    if similarity > highest_similarity and similarity >= threshold:
                        highest_similarity = similarity
                        closest_match_index = index
                return closest_match_index    


            def return_match_label(original_statement, target_sequence, split_statement):
                labels = [0 for _ in range(len(split_statement))]
                matching_idx = find_closest_partial_match_index(target_sequence, split_statement)
                labels[matching_idx] = 1
                return labels         


            def max_2d_index(array_2d):
                # Find indices of the maximum value
                max_indices = torch.argmax(array_2d)

                # Convert the flat index to 2D indices
                max_index = max_indices // array_2d.shape[1], max_indices % array_2d.shape[1]

                return  max_index[0].item(), max_index[1].item()                    


            def get_max_2d_idx(tensor):
                # Get the index of the highest value
                index = torch.argmax(tensor)
                row_index = index // tensor.shape[1]
                col_index = index % tensor.shape[1]

                return row_index.item(), col_index.item()



            def new_func(input_ids_statement, input_ids_label, attention_masks_label, outputs, encoder):
                """
                sample the human labelled concept data with context
                """

                def find_sequence_indices(sequence, target_tensor):
                    indices = []
                    sequence_length = len(sequence)
                    for i in range(len(target_tensor) - sequence_length + 1):
                        if torch.equal(target_tensor[i:i + sequence_length], sequence):
                            indices.append([i, i + sequence_length])
                    return indices[0]

                input_ids_statement = input_ids_statement[0][1:-1]
                annotation_indicies = find_sequence_indices(  input_ids_label[attention_masks_label==1][1:-1], input_ids_statement  )

                # print(annotation_indicies)

                if encoder != None:
                    emb_seq = outputs[0][0][ annotation_indicies[0]: annotation_indicies[1], : ]
                    emb_seq = encoder(emb_seq.unsqueeze(0))
                else:
                    # print(outputs[0].shape)

                    emb_seq = outputs[0][0][ annotation_indicies[0]: annotation_indicies[1], : ].mean(dim=0).unsqueeze(0)

                return emb_seq                    



            data = list()
            if not UNSUPERVISED:
                prototype_embeddings_mean, _ = sample_labels(dfs_train, 4, bert_classifier, encoder, sample_full=True)
            else:
                prototype_embeddings_mean = None

            predictions = list()
            labels = list()
            activations = list()
            # Iterated all examples of one class
            for i in range(len(dfs_test)):
                for j, split in enumerate(['yes', 'no']):
                    data = list()
                    temp_df = dfs_test[i]
                    temp_df = temp_df[temp_df.label==split]

                    for k in range(len(temp_df)):
                        input_ids_label, attention_masks_label         = preprocessing_for_bert([temp_df.iloc[k].text])
                        input_ids_statement, attention_masks_statement = preprocessing_for_bert([temp_df.iloc[k].statement])

                        if not CONTEXT_AWARE:
                            with torch.no_grad():
                                _, outputs = bert_classifier(input_ids_label, attention_masks_label)
                                emb = outputs[0][:, 0, :]

                        else:
                            with torch.no_grad():
                                _, outputs = bert_classifier(input_ids_statement, attention_masks_statement)
                                emb = new_func(input_ids_statement, input_ids_label, attention_masks_label, outputs, encoder)

                        _, acts = pw_net(emb.to(DEVICE), prototype_embeddings_mean)
                        highest_each_row = torch.max(acts, dim=1)[0]
                        max_acts, max_idxs = torch.topk(highest_each_row, k=3)
                        pred = max_idxs.tolist()

                        predictions.append(pred)
                        labels.append( ((i*2)+j) )
                        activations.append(acts.tolist())           



            predictions = np.array(predictions)
            top1_preds = predictions.T[0]

            top3_preds = list()
            for i in range(len(labels)):
                if labels[i] in predictions[i]:
                    top3_preds.append(labels[i])
                else:
                    top3_preds.append(top1_preds[i])

            print("Top 1:", confusion_matrix(top1_preds, labels))
            print("Top 1 Acc:", accuracy_score(top1_preds, labels))
            print("Top 3:", confusion_matrix(top3_preds, labels))
            print("Top 3 Acc:", accuracy_score(top3_preds, labels))

            data.append(predictions)
            data.append(labels)
            data.append(activations)
            data.append(test_acc_data)


            activations = np.array(activations)
            activations = activations.reshape(activations.shape[0], activations.shape[1])

            plt.boxplot(activations)
            plt.ylabel('Prototype Activations')
            plt.xticks(range(1, 9),  ['IV Liable', 'IV not Liable', "IV Def", "IV not Def", 'CV Liable', 'CV not Liable', "CV Def", "CV not Def"], rotation=20, ha='right' )
            plt.savefig('plots/boxplot' + EXPT_TYPE + str(UNSUPERVISED) + str(CONTEXT_AWARE) + '.pdf')
            plt.close()


            print("Time Taken for full iteration:", time.time() - start_time)
            print("\n\n\n\n\n\n\n\n\n\n")

            file_name = 'data/' + EXPT_TYPE + str(UNSUPERVISED) + str(CONTEXT_AWARE)
            with open(file_name + ".dat", "wb") as f:
                pickle.dump(data, f)




