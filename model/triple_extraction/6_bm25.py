from rank_bm25 import BM25Okapi
import json
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainfile', type=str, default='dataset/ade/train.json', help='training file location')
parser.add_argument('--testfile', type=str, default='dataset/ade/test.json', help='training file location')
# parser.add_argument('--train', type=bool, default=True, help='Train instruction True')
parser.add_argument('--train', action='store_true') #with --train true, without false
args = parser.parse_args()

trainfile = args.trainfile
testfile = args.testfile
istrain = args.train


corpus=[]
with open(trainfile) as fr:
    for line in fr.readlines():
        line=json.loads(line.strip())
        for li in line:
            if li["triple_list"][0][1]!="None":
                corpus.append("context: "+li["text"] + "response: "+ "|".join(li["triple_list"][0]))


# instruction = f"You are an excellent linguist. The task is to predict the  relationship between the given head entity and tail entity in a sentence, this relation  must be in ('PREDISPOSES', 'COMPLICATES', 'CAUSES', 'INHIBITS', 'ASSOCIATED_WITH', 'MANIFESTATION_OF', 'PREVENTS', 'INTERACTS_WITH', 'AFFECTS', 'PRODUCES', 'AUGMENTS', 'DISRUPTS', 'STIMULATES', 'COEXISTS_WITH', 'TREATS')"
with open('instruction.txt', 'r') as file:
    instruction = file.read()

tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

if istrain:
    fw=open("train_instruction_container.json", "w")
    _file = trainfile
else:
    fw=open("test_instruction_container.json", "w")
    _file = testfile


h=0
with open(_file) as fr:
    for line in fr.readlines():
        line = json.loads(line.strip())
        for li in line:
            if li["triple_list"][0][1]=="None":
                continue
            h=h+1
            print(h)
            sentence_for_LP = "context: "+li["text"] + "response: "+ "|".join(li["triple_list"][0])
            query=sentence_for_LP
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            max_indices = np.where(doc_scores == np.max(doc_scores))[0].tolist()
            Examples = [corpus[idx] for idx in max_indices]
            Examples = list(set(Examples))
            # Example=bm25.get_top_n(tokenized_query, corpus, n=1)[0]
            Dic_={}
            Dic_["instruction"] =instruction+" "+"Example: "+ " Excample: ".join(Examples)
            Dic_["context"] = li["text"]        
            Dic_["response"] = "|".join(li["triple_list"][0])
            Dic_["category"] = "triplet extraction"


            fw.write(json.dumps(Dic_))
            fw.write("\n")


fw.close()