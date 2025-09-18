# 以《红楼梦》为数据集的GPT-2文本生成模型(学生：陈默涵)

## 项目概述
本项目是复旦大学人工智能设计课程的Entrance-Exam，基于GPT-2模型和《红楼梦》文本语料训练了一个古典文学风格的文本生成模型。通过该项目，本人探索了预训练语言模型在古典文学领域的文本续写能力，展示了机器学习技术与传统文化结合的可能性。

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
贾母向宝玉道：“你这个可是什么？＂凤姐陪笑道：“没什么缘故，他大约是想老太太的意思。”贾母连忙扶了珍珠儿，凤姐也跟着过来。走至半路，正遇王夫人过来，一一回明了贾母。贾母自然又是哀痛的，只因要到宝玉那边，只得忍泪含悲的说道：“既这么着，我也不过去了。由你们办罢，我看着心里也难受，只别委屈了他就是了。”王夫人凤姐一一答应了。贾母才过宝玉这边来，见了宝玉，因问：“你做什么找我？＂宝玉笑道：“我昨日晚上看见林妹妹来了，他说要回南去。我想没人留的住，还得老太太给我留一留他。”贾母听着，说：“使得，只管放心罢。”袭人因扶宝玉躺下。贾母出来到宝钗这边来。那时宝钗尚未回九，所以每每见了人倒有些含羞之意。这一天见贾母满面泪痕，递了茶，贾母叫他坐下。宝钗侧身陪着坐了，才问道：“听得林妹妹病了，不知他可好些了？＂贾母听了这话，那眼泪止不住流下来，因说道：“我的儿，我告诉你，你可别告诉宝玉。都是因你林妹妹，才叫你受了多少委屈。你如今作媳妇了，我才告诉你。这如今你林妹妹没了两三天了，就是娶你的那个时辰死的。如今宝玉这一番病还是为着这个，你们先都在园子里，自然也都是明白的。”宝钗把脸飞红了，想到黛玉之死，又不免落下泪来。贾母又说了一回话去了。自此宝钗千回万转，想了一个主意，只不肯造次，所以过了回九才想出这个法子来。如今果然好些，然后大家说话才不至似前留神。独是宝玉虽然病势一天好似一天，他的痴心总不能解，必要亲去哭他一场。贾母等知他病未除根，不许他胡思乱想，怎奈他郁闷难堪，病多反复。倒是大夫看出心病，索性叫他开散了，再用药调理，倒可好得快些。宝玉听说，立刻要往潇湘馆来。贾母等只得叫人抬了竹椅子过来，扶宝玉坐上。贾母王夫人即便先行。到了潇湘馆内，一见黛玉灵柩，贾母已哭得泪干气绝。凤姐等再三劝住。王夫人也哭了一场。李纨便请贾母王夫人在里间歇着，犹自落泪。宝玉一到，想起未病之先来到这里，今日屋在人亡，不禁嚎啕大哭。想起从前何等亲密，今日死别，怎不更加伤感。众人原恐宝玉病后过哀，都来解劝，宝玉已经哭得死去活来，大家搀扶歇息。其余随来的，如宝钗，俱极痛哭。独是宝玉必要叫紫鹃来见，问明姑娘临死有何话说。紫鹃本来深恨宝玉，见如此，心里已回过来些，又见贾母王夫人都在这里，不敢洒落宝玉，便将林姑娘怎么复病，怎么烧毁帕子，焚化诗稿，并将临死说的话，一一的都告诉了。宝玉又哭得气噎喉干。探春趁便又将黛玉临终嘱咐带柩回南的话也说了一遍。贾母王夫人又哭起来。


## 致谢
- 感谢课程老师提供的GPT-2训练框架代码
- 《红楼梦》文本数据来源于公开电子文献
- 基于Hugging Face Transformers库实现模型训练与推理

## 许可证
本项目仅供课程学习使用，数据集版权归原作者所有。
