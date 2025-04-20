import pandas as pd

def calculate_average_length(texts, is_chinese):
    total_length = 0
    for text in texts:
        if pd.notna(text):
            if is_chinese:
                total_length += len(text)
            else:
                total_length += len(text.split())
    return total_length / len(texts) if len(texts) > 0 else 0

# 读取 CSV 文件
csv_path = 'data/VQA.csv'
df = pd.read_csv(csv_path)

# 1. 每个图片平均有几个问题
average_questions_per_image = df['Image'].value_counts().mean()

# 2. Question 的中文、英文的平均长度
chinese_question_length = calculate_average_length(df['Question'], is_chinese=True)
english_question_length = calculate_average_length(df['Question_en'], is_chinese=False)

# 3. 选项的中文、英文的平均长度
chinese_options = pd.concat([df['A'], df['B'], df['C'], df['D']])
english_options = pd.concat([df['A_en'], df['B_en'], df['C_en'], df['D_en']])
chinese_option_length = calculate_average_length(chinese_options, is_chinese=True)
english_option_length = calculate_average_length(english_options, is_chinese=False)

# 4. Explanation 的中文、英文的平均长度
chinese_explanation_length = calculate_average_length(df['Explanation'], is_chinese=True)
english_explanation_length = calculate_average_length(df['Explanation_en'], is_chinese=False)

# 输出结果，保留两位小数
print(f"每个图片平均的问题数量: {average_questions_per_image:.2f}")
print(f"Question 中文平均长度: {chinese_question_length:.2f} 字")
print(f"Question 英文平均长度: {english_question_length:.2f} 个单词")
print(f"选项中文平均长度: {chinese_option_length:.2f} 字")
print(f"选项英文平均长度: {english_option_length:.2f} 个单词")
print(f"Explanation 中文平均长度: {chinese_explanation_length:.2f} 字")
print(f"Explanation 英文平均长度: {english_explanation_length:.2f} 个单词")