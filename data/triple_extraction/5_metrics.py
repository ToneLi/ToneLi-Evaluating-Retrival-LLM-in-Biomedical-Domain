import json
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--number', type=str, default='5000', help='checkpoint number')
args = parser.parse_args()
number_ = args.number

filename = "test_inference_groundtruth_%s.json"%number_
# def get_ground():
#     all_=[]
#     with open("data/renew_test.json", "r", encoding="utf-8") as fr:
#         for line in fr.readlines():
#             line=line.strip()
#             line=json.loads(line)
#             all_.append([line["SUBJECT_TEXT"],line["PREDICATE"],line["OBJECT_TEXT"]])
#     return all_
        



# ground_gold=get_ground()



def calclulate_f1(statics_dict, prefix=""):
    """
    Calculate the prec, recall and f1-score for the given state dict.
    The state dict contains predict_num, golden_num, correct_num.
    Reutrn a dict in the form as "prefx-recall": 0.99.
    """
    #{'p': 93, 'c': 34, 'g': 320}
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}


# triple extraction
state_dict = {"p": 0, "c": 0, "g": 0}
state_dict_head = {"p": 0, "c": 0, "g": 0}
state_dict_relation = {"p": 0, "c": 0, "g": 0}
state_dict_tail = {"p": 0, "c": 0, "g": 0}
i=-1
with open(filename, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line=line.strip()
        line=json.loads(line)
        i=i+1
        # print(line)
        P=set()
        P_head = set()
        P_relation = set()
        P_tail = set()

        gold_triples=set()
        gold_triples_head = set()
        gold_triples_relation = set()
        gold_triples_tail = set()

        sentence=line["sentence"].lower()
        ground_truth=line["ground_truth"]
        # print(line["ground_truth"]) 
    

        gold_t=ground_truth.split("|")
        gh=gold_t[0].lower().replace("       "," ")
        gr=gold_t[1].lower()
        gt=gold_t[2].lower().replace("       "," ")
      
        gold_triples.add((gh,gr,gt))
        gold_triples_head.add(gh)
        gold_triples_relation.add(gr)
        gold_triples_tail.add(gt)

        if len(line["predicted"].split("\n\n")) < 4:
            continue
        
        predictions=line["predicted"].split("\n\n")[3].split("\n")


        if len(predictions)==2:
            
            predicted_=predictions[1]
            # print(predicted_)
            predicte_t=predicted_.split("|")
            if len(predicte_t)==3:
                ph=predicte_t[0].lower().replace("       "," ").strip()
           
                pt=predicte_t[2].lower().replace("       "," ").strip()#.replace(" .","").replace(" 3. context:{}. response: {}","").replace(" 1. context:{}. response: {}","").replace(" 2. context:{}. response: {}","").replace(" 3. response: {}","")
                pr=predicte_t[1].lower()
             
                P.add((ph,pr,pt))
                P_head.add(ph)
                P_relation.add(pr)
                P_tail.add(pt)
            
   
        state_dict["p"] += len(P)
        state_dict["g"] += len(gold_triples)
        state_dict["c"] += len(P & gold_triples)

        state_dict_head["p"] += len(P_head)
        state_dict_head["g"] += len(gold_triples_head)
        state_dict_head["c"] += len(P_head & gold_triples_head)

        state_dict_relation["p"] += len(P_relation)
        state_dict_relation["g"] += len(gold_triples_relation)
        state_dict_relation["c"] += len(P_relation & gold_triples_relation)

        state_dict_tail["p"] += len(P_tail)
        state_dict_tail["g"] += len(gold_triples_tail)
        state_dict_tail["c"] += len(P_tail & gold_triples_tail)

all_metirc_results = calclulate_f1(state_dict, 'all')
head_metirc_results = calclulate_f1(state_dict_head, 'head')
relation_metirc_results = calclulate_f1(state_dict_relation, 'relation')
tail_metirc_results = calclulate_f1(state_dict_tail, 'tail')


with open('metric_result.txt', 'a') as file:
    # 在文件末尾添加内容
    strw = "checkpoint:%s"%number_
    print(strw)
    file.write(strw)
    file.write("\n")
    file.write(str(all_metirc_results))
    print(all_metirc_results)
    file.write("\n")
    file.write(str(head_metirc_results))
    print(head_metirc_results)
    file.write("\n")
    file.write(str(relation_metirc_results))
    print(relation_metirc_results)
    file.write("\n")
    file.write(str(tail_metirc_results))
    print(tail_metirc_results)
    file.write("\n")
    values = [all_metirc_results['all-prec'],all_metirc_results['all-recall'],all_metirc_results['all-f1'],\
        head_metirc_results['head-prec'], head_metirc_results['head-recall'], head_metirc_results['head-f1'],\
        relation_metirc_results['relation-prec'], relation_metirc_results['relation-recall'], relation_metirc_results['relation-f1'],\
        tail_metirc_results['tail-prec'], tail_metirc_results['tail-recall'], tail_metirc_results['tail-f1']]
    file.write("%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f"% tuple([100*i for i in values]))
    file.write("\n\n")





"""
5000: {'all-prec': 0.7917570498915402, 'all-recall': 0.7849462365591398, 'all-f1': 0.7883369330453565}
8000: {'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}


triple:{'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}
h: {'all-prec': 0.928416485900217, 'all-recall': 0.9204301075268817, 'all-f1': 0.9244060475161987}
t:{'all-prec': 0.9175704989154013, 'all-recall': 0.9096774193548387, 'all-f1': 0.9136069114470842}
r:  {'all-prec': 0.8720173535791758, 'all-recall': 0.864516129032258, 'all-f1': 0.8682505399568036}
"""
