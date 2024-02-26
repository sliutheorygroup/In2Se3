def extract_data_from_dump(filename):
    elements = []
    c_ql1_values = []
    c_ql2_values = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        # 判断是否到了ATOMS部分
        found_atoms_section = False

        for line in lines:
            # 移除首尾空白字符
            line = line.strip()

            if line.startswith("ITEM: ATOMS"):
                found_atoms_section = True
                # 获取列的名称
                column_names = line.split()[2:]
                element_index = column_names.index("element")
                c_ql1_index = column_names.index("c_ql[1]")
                c_ql2_index = column_names.index("c_ql[2]")
            elif found_atoms_section and line:
                # 分割数据
                data = line.split()
                # 获取element和c_ql[1]、c_ql[2]的数值
                element = data[element_index]
                c_ql1_value = float(data[c_ql1_index])
                c_ql2_value = float(data[c_ql2_index])

                elements.append(element)
                c_ql1_values.append(c_ql1_value)
                c_ql2_values.append(c_ql2_value)

    return elements, c_ql1_values, c_ql2_values

def save_to_txt(elements, c_ql1_values, c_ql2_values, output_file):
    with open(output_file, 'w') as file:
        #file.write("Element, c_ql[1], c_ql[2]\n")
        for element, c_ql1_value, c_ql2_value in zip(elements, c_ql1_values, c_ql2_values):
            file.write(f"{element}   {c_ql1_value}   {c_ql2_value}    \n")
 #           print(f"Element: {element}, c_ql[1]: {c_ql1_value}, c_ql[2]: {c_ql2_value}")


# 替换'your_dump_file.xyz'为你的dump文件的实际路径
dump_file = 'dumpql.xyz'
output_file = 'ele_q46.txt'

elements, c_ql1_values, c_ql2_values = extract_data_from_dump(dump_file)
save_to_txt(elements, c_ql1_values, c_ql2_values, output_file)

