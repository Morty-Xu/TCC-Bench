# -*- coding: utf-8 -*-

import json
import re
from collections import Counter
import argparse
import os
from prettytable import PrettyTable

def extract_option_labels(text):
    if isinstance(text, dict):
        return 'error'
    text = text.strip()
    # print(text)
    if text in ["A", "B", "C", "D"]:
        return text
    
    if "\n答案" in text:
        # 取\n答案后的内容
        text = text.split("\n答案")[-1]
        for char in ["A", "B", "C", "D"]:
            if char in text:
                return char
            
    if "\n**答案：" in text:
        # 取\n**答案：后的内容
        text = text.split("\n**答案：")[-1]
        for char in ["A", "B", "C", "D"]:
            if char in text:
                return char
            
    if "\nAnswer" in text:
        # 取\n答案后的内容
        text = text.split("\nAnswer")[-1]
        for char in ["A", "B", "C", "D"]:
            if char in text:
                return char
                
    if "\n**Answer:" in text:
        # 取\n**答案：后的内容
        text = text.split("\n**Answer:")[-1]
        for char in ["A", "B", "C", "D"]:
            if char in text:
                return char
    
    if isinstance(text, dict):
        return 'error'
    pattern = r"\(([A-F])\)"
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r"\b([A-F])\b"
        matches = re.findall(pattern, text)
    # matches = False
    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    # else:
    #     if options:
    #         counter = Counter()
    #         for i, option in enumerate(options, start=1):
    #             label = chr(64 + i)
    #             option_stripped = option.strip()
    #             if option_stripped in text:
    #                 counter[label] += 1
    #             elif text in option:
    #                 counter[label] += 1
    #         if counter:
    #             most_common = counter.most_common()
    #             max_count = most_common[0][1]
    #             candidates = [item for item in most_common if item[1] == max_count]
    #             return candidates[-1][0]
    return None

def calculate_accuracy(file_path, save_dir):
    data = []
    acc = 0
    count = 0
    err = 0
    miss = 0
    category_stats = {}

    with open(file_path, "r") as file:
        for line in file:
            data_ = json.loads(line)
            data.append(data_)
            category = data_["Category"]
            if category not in category_stats:
                category_stats[category] = {"acc": 0, "count": 0, "err": 0, "miss": 0}

    for sample in data:
        category = sample["Category"]
        if sample["response"] != "":
            predict = extract_option_labels(sample["response"])
            sample["extracted_answer"] = predict
            if predict and sample["Answer"] == predict:
                acc += 1
                category_stats[category]["acc"] += 1
                sample["status"] = "correct"
            elif predict is None:
                miss += 1
                category_stats[category]["miss"] += 1
                sample["status"] = "miss"
            elif predict == 'error':
                err += 1
                category_stats[category]["err"] += 1
                sample["status"] = "error"
            else:
                sample["status"] = "incorrect"
        count += 1
        category_stats[category]["count"] += 1

    accuracy = acc / count if count else 0
    errors = err / count if count else 0
    misses = miss / count if count else 0

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    with open(save_path, "w") as file:
        for sample in data:
            json.dump(sample, file, ensure_ascii=False)
            file.write("\n")

    category_results = {}
    for category, stats in category_stats.items():
        if stats["count"] > 0:
            category_results[category] = {
                "accuracy": stats["acc"] / stats["count"],
                "errors": stats["err"] / stats["count"],
                "misses": stats["miss"] / stats["count"]
            }
        else:
            category_results[category] = {"accuracy": 0, "errors": 0, "misses": 0}

    return accuracy, errors, misses, category_results


def evaluate_all_files(output_dir, save_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Language", "Accuracy", "Errors", "Misses"]

    category_tables = {}
    files = sorted(os.listdir(output_dir))
    pattern = r'^([a-zA-Z0-9.\-_]+)_prompt_([a-zA-Z]+).*\.yaml\.jsonl$'

    for file_name in files:
        if file_name.endswith('.jsonl'):
            match = re.match(pattern, file_name)
            print(match)
            if match:
                model_name = match.group(1)
                language = match.group(2)
                file_path = os.path.join(output_dir, file_name)
                accuracy, errors, misses, category_results = calculate_accuracy(file_path, save_dir)
                results.add_row([model_name, language, f"{accuracy:.2%}", f"{errors:.2%}", f"{misses:.2%}"])
                
                for category, stats in category_results.items():
                    if category not in category_tables:
                        category_tables[category] = PrettyTable()
                        category_tables[category].field_names = ["Model", "Language", "Accuracy", "Errors", "Misses"]
                    category_tables[category].add_row([model_name, language, f"{stats['accuracy']:.2%}", f"{stats['errors']:.2%}", f"{stats['misses']:.2%}"])

    print("Overall Results")
    print(results)
    
    for category, table in category_tables.items():
        print(f"\nCategory: {category}")
        print(table)

def main(args):
    evaluate_all_files(args.output_dir, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy.")
    parser.add_argument('--output_dir', type=str, default='results_tcc', help='Directory to read result files from')
    parser.add_argument('--save_dir', type=str, default='results_statistic_tcc', help='Directory to save result files with category')
    
    args = parser.parse_args()
    main(args)
