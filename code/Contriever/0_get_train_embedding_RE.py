import torch
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np


sentences=[]
with open("final_all_zero_shot_train.json") as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        sentences.append("context: In sentence "+line["sentence"]+"the relationship between "+ line["head"]+" and "+line["tail"]+"is ? "+"response: "+line["relation"])


tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')


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
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    outputs = model(**inputs)
    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    np_embedding=embeddings.detach().numpy()[0]
    all_sentence_vector.append(np_embedding)


np.save("train_embedding_RE.npy", all_sentence_vector)
