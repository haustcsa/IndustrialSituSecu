

def clustering(index1,index2):
    file_path = "F:\getModbus0716.txt"
    line_categories = {}  # 使用字典存储按行长度分类的结果

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # 去除行首尾的空格或换行符
            line_length = len(line)

            # 将行添加到相应长度的分类中
            if line_length in line_categories:
                line_categories[line_length].append(line)
            else:
                line_categories[line_length] = [line]


        for key, value in line_categories.items():
            sequences = line_categories[key]

            classified_sequences = {}
            for seq in sequences:
                func_code = seq[14:16]
                if func_code not in classified_sequences:
                    classified_sequences[func_code] = []
                classified_sequences[func_code].append(seq)
                line_categories[key] = classified_sequences


    return line_categories[index1][index2]

def writeFile(data,file_path):

    with open(file_path,'w') as file:
        for line in data:
            file.writelines(line+'\n')

def changeStart(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for i in range(len(lines)):
        new_line = '{:04X}'.format(i) + lines[i][4:]
        new_lines.append(new_line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    file_path1 = "F:\getModbus"+"2403"+".txt"
    writeFile(clustering(24,'03'),file_path1)
    changeStart(file_path1)
    file_path2 = "F:\getModbus" + "2406" + ".txt"
    writeFile(clustering(24, '06'), file_path2)
    changeStart(file_path2)
    file_path3 = "F:\getModbus" + "2410" + ".txt"
    writeFile(clustering(24, '10'), file_path3)
    changeStart(file_path3)
    file_path4 = "F:\getModbus" + "6610" + ".txt"
    writeFile(clustering(66, '10'), file_path4)
    changeStart(file_path4)
    file_path5 = "F:\getModbus" + "7417" + ".txt"
    writeFile(clustering(74, '17'), file_path5)
    changeStart(file_path5)




