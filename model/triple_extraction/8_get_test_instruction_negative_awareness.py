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
parser.add_argument('--testfile', type=str, default='dataset/ade/test.json', help='training file location')
parser.add_argument('--triever', type=str, default='facebook/contriever', help='retriver name')
args = parser.parse_args()

trainfile = args.trainfile
testfile = args.testfile
triever = args.triever

sentences=[]
with open(trainfile) as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        for li in line:
            if li["triple_list"][0][1]!="None":
                sentences.append("context: "+li["text"]) #+ "response: "+ "|".join(li["triple_list"][0]))

                
Stored_Embeddings=np.load("train_embedding.npy")


# tokenizer = AutoTokenizer.from_pretrained(triever)
# model = AutoModel.from_pretrained(triever)
tokenizer, model = model_creator(triever)


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

# Mean pooling

# Apply tokenizer
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

# instruction = f"You are an excellent linguist. The task is to predict the  relationship between the given head entity and tail entity in a sentence, this relation  must be in ('PREDISPOSES', 'DIAGNOSES', 'INTERACTS_WITH', 'ADMINISTERED_TO', 'ASSOCIATED_WITH', 'STIMULATES', 'AFFECTS', 'PREVENTS', 'USES', 'CAUSES', 'TREATS', 'PROCESS_OF')"
# instruction = f"please extract the triplet from this sentence, the triplet is [head entity, relation, tail entity], \
# the element relation denotes the relationship between head entity and tail entity, I will provide you \
# the definition of the triplet you need to extract, the sentence from where your extract the triplets \
# (head entity, relation, tail_entity) and the output format with examples. the relation must in my \
# predefined relation set: ('effect', 'advise', 'mechanism', 'int').  \
# response Format: head entity|relation|tail entity."
with open('instruction_negative_awareness.txt', 'r') as file:
    # Read the content of the file
    instruction = file.read()


fw=open("test_instruction_container.json", "w")

h=0
with open(testfile) as fr:
    for line in fr.readlines():
        line = json.loads(line.strip())
        for li in line:
            if li["triple_list"][0][1]=="None":
                continue
            h=h+1
            print(h)
            sentence="context: " + li["text"]
            inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
                # Compute token embeddings
            outputs = model(**inputs)
            # Mean pooling
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            np_embedding=embeddings.detach().numpy()[0]
            all_cos=[]
            for i in range(len(Stored_Embeddings)):
                cos_=get_cos_similar(np_embedding,Stored_Embeddings[i])
                all_cos.append(cos_)
                
            arr = np.array(all_cos)
#             max_=arr.argsort()[-1:][::-1][0]
            max_indices = np.where(arr == np.max(arr))[0].tolist()
            Examples = [sentences[idx] for idx in max_indices]
            Examples = list(set(Examples))

#             Excample=sentences[max_index]
            # print(excamples_)
            # print(excamples_)


            Dic_ = {}
            Dic_["instruction"] = instruction #+ " retrieved sentence: " + " ".join(Examples)
            Dic_["context"] = "retrieved sentence: " + " ".join(Examples)  #li["text"]
            Dic_["response"] = "|".join(li["triple_list"][0])

            Dic_["category"] = "triplet extraction"

            fw.write(json.dumps(Dic_))
            fw.write("\n")
#             if len(Examples)>1:
#                 print()
#                 print()
#                 print()
#                 print()
#                 print()
#                 print()
#                 print()
            # print(Dic_)
            # print()

fw.close()