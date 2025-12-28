# # import pandas as pd
# # import re
# # from clean_gadget import clean_gadget
# #
# #
# # def normalization(source):
# #     nor_code = []
# #     for fun in source['code']:
# #         lines = fun.split('\n')
# #         # print(lines)
# #         code = ''
# #         for line in lines:
# #             line = line.strip()
# #             line = re.sub('//.*', '', line)
# #             code += line + ' '
# #         # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
# #         code = re.sub('/\\*.*?\\*/', '', code)
# #         code = clean_gadget([code])
# #         nor_code.append(code[0])
# #         print(code[0])
# #     return nor_code
# #
# #
# # def mutrvd():
# #     train = pd.read_pickle('trvd_train.pkl')
# #     test = pd.read_pickle('trvd_test.pkl')
# #     val = pd.read_pickle('trvd_val.pkl')
# #
# #     train['code'] = normalization(train)
# #     train.to_pickle('./mutrvd/train.pkl')
# #
# #     test['code'] = normalization(test)
# #     test.to_pickle('./mutrvd/test.pkl')
# #
# #     val['code'] = normalization(val)
# #     val.to_pickle('./mutrvd/val.pkl')
# #
# #
# # if __name__ == '__main__':
# #     mutrvd()
#
#
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import re
# from clean_gadget import clean_gadget
#
# # Step 1: Create the directories if they do not exist
# os.makedirs('dataset/mutrvd', exist_ok=True)
# os.makedirs('subtrees/mutrvd', exist_ok=True)
# os.makedirs('mutrvd', exist_ok=True)  # Ensure the 'mutrvd' directory exists
#
# # Step 2: Load the dataset.pkl file
# dataset = pd.read_pickle('dataset/dataset.pkl')
#
# # Step 3: Split the dataset into training, testing, and validation sets
# train, temp = train_test_split(dataset, test_size=0.4, random_state=42)
# test, val = train_test_split(temp, test_size=0.5, random_state=42)
#
# # Step 4: Save the split datasets into the dataset/mutrvd directory
# train.to_pickle('dataset/mutrvd/train.pkl')
# test.to_pickle('dataset/mutrvd/test.pkl')
# val.to_pickle('dataset/mutrvd/val.pkl')
#
# # Step 5: Normalize the data and save to mutrvd directory
# def normalization(source):
#     nor_code = []
#     for fun in source['code']:
#         lines = fun.split('\n')
#         code = ''
#         for line in lines:
#             line = line.strip()
#             line = re.sub('//.*', '', line)  # Remove inline comments
#             code += line + ' '
#         code = re.sub('/\\*.*?\\*/', '', code)  # Remove block comments
#         code = clean_gadget([code])  # Clean code
#         nor_code.append(code[0])
#         print(code[0])
#     return nor_code
#
# def mutrvd():
#     train = pd.read_pickle('dataset/mutrvd/train.pkl')
#     test = pd.read_pickle('dataset/mutrvd/test.pkl')
#     val = pd.read_pickle('dataset/mutrvd/val.pkl')
#
#     train['code'] = normalization(train)
#     train.to_pickle('./mutrvd/train.pkl')
#
#     test['code'] = normalization(test)
#     test.to_pickle('./mutrvd/test.pkl')
#
#     val['code'] = normalization(val)
#     val.to_pickle('./mutrvd/val.pkl')
#
# if __name__ == '__main__':
#     mutrvd()
#

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from clean_gadget import clean_gadget

# Step 1: Create the directories if they do not exist
os.makedirs('dataset/mutrvd', exist_ok=True)
os.makedirs('subtrees/mutrvd', exist_ok=True)
os.makedirs('mutrvd', exist_ok=True)  # Ensure the 'mutrvd' directory exists

# Step 2: Load the dataset.pkl file
dataset = pd.read_pickle('dataset/dataset.pkl')


# Step 3: Custom sampling with oversampling for labels 1-85
def custom_sample_dataset(df):
    sampled_df = pd.DataFrame()
    label_counts = df['label'].value_counts()

    for label in range(86):  # 0 to 85 labels
        label_data = df[df['label'] == label]
        if label == 0:
            # For label 0, sample 5000 (or all if less than 5000)
            sample_size = min(5000, len(label_data))
            sampled_label_data = label_data.sample(n=sample_size, random_state=42) if sample_size > 0 else label_data
        else:
            # For labels 1-85, ensure exactly 50 samples with replacement if needed
            sample_size = 50
            if len(label_data) >= 50:
                sampled_label_data = label_data.sample(n=50, random_state=42)
            else:
                # Oversample with replacement if less than 50
                sampled_label_data = label_data.sample(n=50, replace=True, random_state=42)

        sampled_df = pd.concat([sampled_df, sampled_label_data])

    return sampled_df


# Step 4: Apply custom sampling and split the dataset
sampled_dataset = custom_sample_dataset(dataset)
train, temp = train_test_split(sampled_dataset, test_size=0.4, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

# Step 5: Save the split datasets into the dataset/mutrvd directory
train.to_pickle('dataset/mutrvd/train.pkl')
test.to_pickle('dataset/mutrvd/test.pkl')
val.to_pickle('dataset/mutrvd/val.pkl')


# Step 6: Normalize the data and save to mutrvd directory
def normalization(source):
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)  # Remove inline comments
            code += line + ' '
        code = re.sub('/\\*.*?\\*/', '', code)  # Remove block comments
        code = clean_gadget([code])  # Clean code
        nor_code.append(code[0])
        print(code[0])
    return nor_code


def mutrvd():
    train = pd.read_pickle('dataset/mutrvd/train.pkl')
    test = pd.read_pickle('dataset/mutrvd/test.pkl')
    val = pd.read_pickle('dataset/mutrvd/val.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')


if __name__ == '__main__':
    # 调试：检查采样后的标签分布
    sampled_dataset = custom_sample_dataset(dataset)
    print("Sampled dataset label distribution:\n", sampled_dataset['label'].value_counts())
    mutrvd()
