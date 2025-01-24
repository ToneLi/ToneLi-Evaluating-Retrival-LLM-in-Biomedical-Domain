from rank_bm25 import BM25Okapi
import json
def make_corpus():
    corpus=[]
    corpus_file=r"D:\2_my_project\0_one_for_all\0_AZ\AZ_link_prediction_train.json"
    with  open(corpus_file) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())
            sentence_for_LP="Example: Context: in sentence "+line["sentence"]+" the relationship between "+line["head"]+" and "+line["tail"]+" is ? Response: "+line["relation"]
            corpus.append(sentence_for_LP)
    return corpus

instruction = f"You are an excellent linguist. The task is to predict the  relationship between the given head entity and tail entity in a sentence, this relation  must be in ('PREDISPOSES', 'COMPLICATES', 'CAUSES', 'INHIBITS', 'ASSOCIATED_WITH', 'MANIFESTATION_OF', 'PREVENTS', 'INTERACTS_WITH', 'AFFECTS', 'PRODUCES', 'AUGMENTS', 'DISRUPTS', 'STIMULATES', 'COEXISTS_WITH', 'TREATS')"


corpus=make_corpus()
# corpus = [
#     "Hello there good man!",
#     "It is quite windy in London",
#     "How is the weather today?"
# ]
#
tokenized_corpus = [doc.split(" ") for doc in corpus]
#
bm25 = BM25Okapi(tokenized_corpus)
#
fw=open("AZ_RE_test_instruction.json", "w")

_file = r"D:\2_my_project\0_one_for_all\0_AZ\AZ_link_prediction_test.json"

# relations=[]
i=0
with  open(_file) as fr:
    for line in fr.readlines():
        i=i+1
        print(i)
        line = json.loads(line.strip())
        # relations.append(line["relation"])
        sentence_for_LP = "Example: Context: in sentence "+ line["sentence"] +" the relationship between " + line["head"] + " and " + line[
            "tail"] + " is ? Response: " + line["relation"]
        query=sentence_for_LP
        tokenized_query = query.split(" ")
        Excample=bm25.get_top_n(tokenized_query, corpus, n=1)[0]
        Dic_={}
        Dic_["instruction"] =instruction+" "+Excample
        Dic_["context"] ="In sentence "+ line["sentence"] +" the relationship between " + line["head"] + " and " + line["tail"] + " is ?"
        Dic_["response"] = line["relation"]
        # print( Dic_["response"])

        Dic_["category"] ="AZ_relation_extraction_bm25"

        fw.write(json.dumps(Dic_))
        fw.write("\n")



# print(set(relations))