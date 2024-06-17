from sklearn.metrics import precision_score, recall_score, f1_score
import json
import re

number_="5000"


file_name = "prediction/hetionet5k_lp_test_instruction_link_prediction%s.json"%number_


preds = []
labels = []

with open(file_name, "r", encoding="utf-8") as fr:  # llama2_7b_1000_KNN_0.json   llama2_7b_sample_1000.json
    for line in fr.readlines():
        line=line.strip()
        line=json.loads(line)
        P=set()
        sentence=line["sentence"].lower()
        gold_triples=set()
        ground_truth=line["ground_truth"]

        predictions=line["predicted"].split("\n\n")[3].split("\n")[1].split(" ")[0]
        # predictions=line["predicted"].split("\n\n")[3].split("\n")[1]

        preds.append(predictions)
        labels.append(ground_truth)

y_true = labels
y_pred = preds

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Macro Precision: {precision:.6f}")
print(f"Macro Recall: {recall:.6f}")
print(f"Macro F1 Score: {f1:.6f}")

precision_micro = precision_score(y_true, y_pred, average='micro')
recall_micro = recall_score(y_true, y_pred, average='micro')
f1_micro = f1_score(y_true, y_pred, average='micro')

print(f"Micro Precision: {precision_micro:.6f}")
print(f"Micro Recall: {recall_micro:.6f}")
print(f"Micro F1 Score: {f1_micro:.6}")