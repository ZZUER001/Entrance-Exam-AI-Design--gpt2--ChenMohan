# 《红楼梦》GPT-2文本生成模型(学生：陈默涵)

## 项目概述
本项目是复旦大学人工智能设计课程的Entrance-Exam，基于GPT-2模型和《红楼梦》文本语料训练了一个古典文学风格的文本生成模型。通过该项目，本人探索了预训练语言模型在古典文学领域的文本续写能力，展示了机器学习技术与传统文化结合的可能性。

### 核心特点
- 使用GPT-2小型模型（12层，隐藏维度768）进行训练
- 基于《红楼梦》全本语料进行微调
- 支持自定义长度和温度参数的文本生成
- 保留古典文学的语言风格和人物关系特征

## 环境要求
- Python 3.7+
- PyTorch 1.7+
- Transformers 4.0+
- CUDA 10.1+（使用GPU训练）

### 安装依赖
pip install torch transformers numpy tqdm
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

## 项目流程

### 1. 数据准备
将《红楼梦》文本处理为JSON格式，存放于`data/train.json`，格式要求：
["第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀...", "第二回 贾夫人仙逝扬州城 冷子兴演说荣国府...", ...]

### 2. 数据预处理
运行数据预处理脚本，将文本转换为模型输入格式：
python train.py --raw --tokenizer_path cache/vocab_small.txt \
                --raw_data_path data/train.json \
                --tokenized_data_path data/tokenized/ \
                --num_pieces 100 --min_length 128

### 3. 模型训练
python train.py --device 0 --epochs 100 --batch_size 8 \
                --lr 1.5e-4 --warmup_steps 2000 \
                --output_dir model/ --log_step 10

### 4. 文本生成
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "model/final_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
model.to("cuda")

prompt = "贾母向宝玉道"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_length=200,
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

## 示例输出
**提示词**：贾母向宝玉道

**生成文本**：
贾母向宝玉道：“你这个可是什么？＂凤姐陪笑道：“没什么缘故，他大约是想老太太的意思。”贾母连忙扶了珍珠儿，凤姐也跟着过来．走至半路，正遇王夫人过来，一一回明了贾母．贾母自然又是哀痛的，只因要到宝玉那边，只得忍泪含悲的说道：“既这么着，我也不过去了．由你们办罢，我看着心里也难受，只别委屈了他就是了。”王夫人凤姐一一答应了．贾母才过宝玉这边来，见了宝玉，因问：“你做什么找我？＂宝玉笑道：“我昨日晚上看见林妹妹来了，他说要回南去．我想没人留的住，还得老太太给我留一留他。”贾母听着，说：“使得，只管放心罢。”


## 致谢
- 感谢课程老师提供的GPT-2训练框架代码
- 《红楼梦》文本数据来源于公开电子文献
- 基于Hugging Face Transformers库实现模型训练与推理

## 许可证
本项目仅供课程学习使用，数据集版权归原作者所有。
