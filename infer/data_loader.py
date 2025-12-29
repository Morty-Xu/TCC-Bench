import json
import yaml
import os
import random
import pandas as pd
random.seed(42)

# Read the YAML template
def read_yaml(config='default'):
    with open(f'config/{config}', 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
    

# Load the data
def load_data(prompt_template='prompt_zh.yaml', file_path='file_path.yaml', few_shot=None, domain_aware=False):
    template = read_yaml('prompt/' + prompt_template)
    data_path = read_yaml(file_path)
    df = pd.read_csv(data_path['data_path'])

    def format_question(row, zh=True):

        domain_name = {
            'Astronomy': '天文',
            'Music': '音乐',
            'Custom': '民俗',
            'Architecture': '建筑',
            'Transportation': '交通',
            'Diet': '饮食',
            'Clothing': '服装',
            'Artifact': '器物'
        }

        if zh:
            if domain_aware:
                prompt_format = [domain_name[row['Category']], row['Question'], row['A'], row['B'], row['C'], row['D']]
            else:
                prompt_format = [row['Question'], row['A'], row['B'], row['C'], row['D']]
        else:
            prompt_format = [row['Question_en'], row['A_en'], row['B_en'], row['C_en'], row['D_en']]
            
        return template['instruction'] + "\n" + template['prompt_format'][0].format(*prompt_format)

    if few_shot is None:
        # Zero-shot
        for _, row in df.iterrows():
            zh = prompt_template in ['prompt_zh.yaml', 'prompt_CoT.yaml', 'prompt_zh_exp.yaml', 'prompt_zh_domain_aware.yaml']
            question = format_question(row, zh)
            image_path = data_path['image_root'] + "/" + str(row['Image'])
            prompt = {'prompt': question, 'image': [image_path], 'id': row['id']}
            yield prompt, row.to_dict()
    else:
        # Few-shot
        df_dict = df.to_dict(orient='records')
        examples = [exa for exa in df_dict if exa["id"] in [1, 341, 659]]
        samples = [s for s in df_dict if s["id"] not in [1, 24, 341]]

        for sample in samples:
            if few_shot == 'one-shot':
                shots = examples[:1]
            elif few_shot == 'three-shot':
                shots = examples
            else:
                shots = []

            conversations = []
            all_instances = shots + [sample]
            for turn_id, instance in enumerate(all_instances):
                zh = prompt_template in ['prompt_zh.yaml', 'prompt_CoT.yaml', 'prompt_zh_exp.yaml']
                question = format_question(instance, zh)
                image_path = data_path['image_root'] + "/" + instance['Image']

                prompt_dict = {
                    'prompt': question,
                    'image': [image_path],
                    'id': f"{sample['id']}-turn-{turn_id}"
                }

                if turn_id < len(all_instances) - 1:
                    prompt_dict["response"] = instance.get("Answer", "A")
                conversations.append(prompt_dict)

            few_shot_prompt = {
                "id": sample["id"],
                "few-shot": True,
                "conversations": conversations
            }
            yield few_shot_prompt, sample


if __name__ == '__main__':
    # df = pd.read_csv(FILE_PATH)
    for prompt, sample in load_data():
        print(prompt['image'])
        break;
    # load_data()