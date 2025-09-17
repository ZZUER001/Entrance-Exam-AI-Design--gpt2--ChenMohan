def replace_halfwidth_to_fullwidth(input_file, output_file):
    """
    将TXT文件中的半角符号替换为全角符号
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    # 半角符号到全角符号的映射表
    half_to_full = {
        # 标点符号
        ',': '，', '.': '。', '?': '？', '!': '！', ';': '；', ':': '：',
        '"': '＂', "'": '＇', '(': '（', ')': '）', '[': '［', ']': '］',
        '{': '｛', '}': '｝', '<': '＜', '>': '＞','．': '。',
        
        # 运算符与特殊符号
        '/': '／', '\\': '＼', '+': '＋', '-': '－', '*': '＊', '%': '％',
        '=': '＝', '#': '＃', '$': '＄', '&': '＆', '@': '＠', '^': '＾',
        '`': '｀', '~': '～', '|': '｜', '_': '＿',
        
        # 空格（半角空格转全角空格）
        ' ': ' '
    }
    
    # 创建字符转换表
    translation_table = str.maketrans(half_to_full)
    
    try:
        # 读取输入文件（指定UTF-8编码，可根据文件实际编码修改）
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 执行符号替换
        processed_content = content.translate(translation_table)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"✅ 转换完成！结果已保存至：{output_file}")
        
    except FileNotFoundError:
        print(f"❌ 错误：输入文件不存在 - {input_file}")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")


if __name__ == "__main__":
    import sys
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法：python half2full.py <输入文件路径> <输出文件路径>")
        print("示例：python half2full.py input.txt output.txt")
        sys.exit(1)
    
    # 获取文件路径并执行转换
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    replace_halfwidth_to_fullwidth(input_path, output_path)
