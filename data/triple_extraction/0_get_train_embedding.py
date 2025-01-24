import sys
sys.path.append('utilities')
import torch
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np
from model_creator import model_creator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', type=str, default='dataset/ade/train.json', help='training file location')
parser.add_argument('--triever', type=str, default='facebook/contriever', help='retriver name')
args = parser.parse_args()

trainfile = args.trainfile
triever = args.triever


sentences=[]
with open(trainfile) as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        for li in line:
            if li["triple_list"][0][1]!="None":
                sentences.append("context: "+li["text"] + ". response: "+ "|".join(li["triple_list"][0]))

# with open(trainfile) as fr:
#     for line in fr.readlines():
#         line=json.loads(line.strip())
#         for li in line:
#             sentences.append("context: "+li["text"]+ "response: "+ li['label'] )


# tokenizer = AutoTokenizer.from_pretrained(triever)
# model = AutoModel.from_pretrained(triever)
tokenizer, model = model_creator(triever)

# Apply tokenizer
# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
# outputs = model(**inputs)


# Mean pooling
# Apply tokenizer
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings



all_sentence_vector=[]
l=-1
for sentence in sentences:
    l=l+1
    print(l)
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Compute token embeddings
    outputs = model(**inputs)
    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    np_embedding=embeddings.detach().numpy()[0]
    all_sentence_vector.append(np_embedding)


np.save("train_embedding.npy", all_sentence_vector)
