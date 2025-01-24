import torch
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')

sentences=[]
with open("final_all_zero_shot_train.json") as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        sentences.append("context: In sentence "+line["sentence"]+"the relationship between "+ line["head"]+" and "+line["tail"]+"is ? "+"response: "+line["relation"])

Stored_Embeddings=np.load("instructor_train_embedding_RE.npy")




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

# instruction = f"You are an excellent linguist. The task is to predict the  relationship between the given head entity and tail entity in a sentence, this relation  must be in ('PREDISPOSES', 'COMPLICATES', 'CAUSES', 'INHIBITS', 'ASSOCIATED_WITH', 'MANIFESTATION_OF', 'PREVENTS', 'INTERACTS_WITH', 'AFFECTS', 'PRODUCES', 'AUGMENTS', 'DISRUPTS', 'STIMULATES', 'COEXISTS_WITH', 'TREATS')"

instruction = f"You are an excellent linguist. The task is to predict the  relationship between the given head entity and tail entity in a sentence, this relation  must be in ('PREVENTS', 'INTERACTS_WITH', 'ASSOCIATED_WITH', 'STIMULATES', 'PREDISPOSES', 'AFFECTS', 'CAUSES', 'TREATS')"

fw=open("AZ_RE_test_instruction_instructor.json", "w")

h=0
with open("AZ_test_zero_shot.json") as fr:
    for line in fr.readlines():
        h=h+1
        print(h)
        line = json.loads(line.strip())
        sentence="context: " + line["sentence"]
        customized_embeddings = model.encode(sentence)

        all_cos=[]
        for i in range(len(Stored_Embeddings)):
            cos_=get_cos_similar(customized_embeddings,Stored_Embeddings[i])
            all_cos.append(cos_)
        arr = np.array(all_cos)
        max_=arr.argsort()[-1:][::-1][0]
        Excample=sentences[max_]
        # print(excamples_)


        Dic_ = {}
        Dic_["instruction"] = instruction + " Excamples: " + Excample
        Dic_["context"] = "In sentence "+ line["sentence"]+" the relationship between " + line["head"] + " and " + line["tail"] + " is ?"
        Dic_["response"] = line["relation"]
        # print( Dic_["response"])

        Dic_["category"] = "AZ_RE_container"

        fw.write(json.dumps(Dic_))
        fw.write("\n")
