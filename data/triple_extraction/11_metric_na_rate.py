import json
import re

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--number', type=str, default='5000', help='checkpoint number')
# args = parser.parse_args()
# number_ = args.number

filename1 = "true_false_test_inference_true_noise.json"
filename2 = "true_false_test_inference_fake_noise.json"

# ground_gold=get_ground()

def awareness_rate(filename, word):
    total=0
    num_tf = 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip()
            line=json.loads(line)

            total += 1

            result = line["predicted"].split("\n\n### Response: \n")[1].split('\n\n')[0]
            if result == word:
                num_tf += 1
    return num_tf/total


true_awareness_rate = awareness_rate(filename1, 'True')
fake_awareness_rate = awareness_rate(filename1, 'False')


with open('metric_result.txt', 'a') as file:
    # 在文件末尾添加内容
    print('true_awareness_rate:', true_awareness_rate)
    print('fake_awareness_rate:', fake_awareness_rate)
    file.write('true_awareness_rate:')
    file.write(str(true_awareness_rate))
    file.write("\n")
    file.write('fake_awareness_rate:')
    file.write(str(fake_awareness_rate))
    file.write("\n")




"""
5000: {'all-prec': 0.7917570498915402, 'all-recall': 0.7849462365591398, 'all-f1': 0.7883369330453565}
8000: {'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}


triple:{'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}
h: {'all-prec': 0.928416485900217, 'all-recall': 0.9204301075268817, 'all-f1': 0.9244060475161987}
t:{'all-prec': 0.9175704989154013, 'all-recall': 0.9096774193548387, 'all-f1': 0.9136069114470842}
r:  {'all-prec': 0.8720173535791758, 'all-recall': 0.864516129032258, 'all-f1': 0.8682505399568036}
"""
