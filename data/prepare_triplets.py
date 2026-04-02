import json
import os
import random
import re
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. 定义各类别的属性词典 (Attribute Dictionaries)
# -----------------------------------------------------------------------------
# 这些词典用于在 prompt 中精准定位 target_word，并用于构造 Negative 样本。
ATTR_DICTS = {
    "color": [
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 
        'orange', 'brown', 'grey', 'gray', 'silver', 'gold', 'cyan', 'magenta'
    ],
    "counting": [
        'a', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'dozen', 'couple', 'few', 'several'
    ],
    "spatial": [
        'left', 'right', 'above', 'below', 'top', 'bottom', 'front', 'behind',
        'inside', 'outside', 'under', 'over', 'next to', 'beside', 'near'
    ],
    "texture": [
        'wooden', 'metallic', 'fluffy', 'glass', 'plastic', 'leather', 'ceramic', 
        'furry', 'smooth', 'rough', 'soft', 'hard', 'shiny', 'matte', 'fuzzy', 'silky'
    ],
    "shape": [
        'round', 'square', 'triangular', 'rectangular', 'oval', 'spherical', 
        'cylindrical', 'circular', 'cube', 'pyramid', 'cone'
    ],
    "scene": [
        'beach', 'forest', 'city', 'street', 'room', 'kitchen', 'bedroom', 
        'bathroom', 'park', 'mountain', 'desert', 'ocean', 'river', 'lake', 
        'office', 'restaurant', 'cafe', 'hospital', 'school', 'library'
    ],
    "non-spatial": [
        # 针对 non-spatial (通常是动作或状态)，提取一些常见的动词/状态词
        'running', 'walking', 'sitting', 'standing', 'sleeping', 'eating', 
        'drinking', 'playing', 'jumping', 'flying', 'swimming', 'smiling', 
        'crying', 'laughing', 'reading', 'writing', 'talking', 'listening'
    ]
}

# -----------------------------------------------------------------------------
# 2. 核心处理逻辑
# -----------------------------------------------------------------------------
def process_task(task_name, input_file, output_file, num_pos_per_anchor: int = 1):
    """
    处理单个任务的 JSONL 文件，生成对比学习三元组。
    """
    if not os.path.exists(input_file):
        print(f"⚠️ Warning: Input file not found -> {input_file}")
        return

    attributes = ATTR_DICTS.get(task_name, [])
    if not attributes:
        print(f"⚠️ Warning: No attribute dictionary defined for task -> {task_name}")
        return

    # 步骤 A: 读取原始数据
    raw_prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get("prompt", "").lower().strip()
                if prompt:
                    raw_prompts.append(prompt)
            except json.JSONDecodeError:
                continue

    # 步骤 A+: counting 任务专属扩增
    # T2I-CompBench 的 counting_metadata.jsonl 只有 one/two/three，
    # 这里通过替换数字词把数据扩充到 four～ten，覆盖更广的数字范围。
    if task_name == "counting":
        EXTEND_NUMBERS = ['four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        BASE_NUMBERS   = ['one', 'two', 'three']
        extra = []
        for prompt in raw_prompts:
            for base in BASE_NUMBERS:
                if re.search(rf'\b{base}\b', prompt):
                    for tgt in EXTEND_NUMBERS:
                        extra.append(re.sub(rf'\b{base}\b', tgt, prompt, count=1))
                    break
        raw_prompts = raw_prompts + extra
        print(f"  [counting] augmented: {len(raw_prompts) - len(extra)} -> {len(raw_prompts)} prompts")

    # 步骤 B: 按 target_word 分组
    # grouped_prompts: { "red": ["a red car", "the red apple"], "blue": [...] }
    grouped_prompts = defaultdict(list)
    for prompt in raw_prompts:
        found_attr = None
        for attr in attributes:
            # 使用 \b 确保匹配完整单词（'red' 不匹配 'tired'）
            if re.search(rf'\b{attr}\b', prompt):
                found_attr = attr
                break
        if found_attr:
            grouped_prompts[found_attr].append(prompt)

    # 步骤 C: 构造三元组 (Anchor, Positive, Negative)
    triplets = []
    
    for target_word, prompts in grouped_prompts.items():
        # 如果该属性词下只有 1 个句子，无法通过组内采样构造 Positive，跳过
        if len(prompts) < 2:
            continue
            
        for anchor_prompt in prompts:
            available_positives = [p for p in prompts if p != anchor_prompt]
            if not available_positives:
                continue
            other_attrs = [a for a in attributes if a != target_word]
            if not other_attrs:
                continue

            # 为每个 anchor 生成 num_pos_per_anchor 个不同的 (positive, negative) 对
            # 这样数据量少的任务（如 scene/texture）可以通过 k > 1 扩增数据
            k = min(num_pos_per_anchor, len(available_positives))
            sampled_positives = random.sample(available_positives, k)

            for positive_prompt in sampled_positives:
                neg_word = random.choice(other_attrs)
                negative_prompt = re.sub(rf'\b{target_word}\b', neg_word, anchor_prompt, count=1)
                if negative_prompt != anchor_prompt:
                    triplets.append({
                        "task": task_name,
                        "target_word": target_word,
                        "anchor": anchor_prompt,
                        "positive": positive_prompt,
                        "negative": negative_prompt
                    })

    # 步骤 D: 保存结果
    if triplets:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for t in triplets:
                f.write(json.dumps(t) + "\n")
        print(f"✅ [{task_name.upper()}] Generated {len(triplets)} triplets -> {output_file}")
    else:
        print(f"⚠️ [{task_name.upper()}] No valid triplets generated.")

# -----------------------------------------------------------------------------
# 3. 主执行入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--in_dir',  type=str, default="/Users/gaomingju/Desktop/code/Z-Image-ours/data/text_prompts")
    _parser.add_argument('--out_dir', type=str, default="/Users/gaomingju/Desktop/code/Z-Image-ours/data/train_triplets")
    # 数据少的任务（scene/texture）可以设大一点，比如 5~10，达到和 counting 相近的数量
    _parser.add_argument('--num_pos_per_anchor', type=int, default=1,
                         help='Number of positive samples per anchor. '
                              'Set > 1 to augment data for low-resource tasks (e.g. scene, texture).')
    _args = _parser.parse_args()

    BASE_IN_DIR  = _args.in_dir
    BASE_OUT_DIR = _args.out_dir
    NUM_POS      = _args.num_pos_per_anchor

    # 每个任务可以单独设置扩增倍数：数据少的任务扩增更多
    TASK_NUM_POS = {
        "color":       NUM_POS,
        "counting":    NUM_POS,
        "spatial":     max(NUM_POS, 2),
        "texture":     max(NUM_POS, 4),
        "shape":       NUM_POS,
        "scene":       max(NUM_POS, 8),   # scene 只有 ~286 条，扩增 8 倍达到 ~2000
        "non-spatial": max(NUM_POS, 3),
    }

    print("🚀 Starting triplet generation for Contrastive Learning...")
    print(f"   num_pos_per_anchor (base) = {NUM_POS}")
    print("-" * 50)

    total_triplets = 0
    for task, k in TASK_NUM_POS.items():
        in_file  = os.path.join(BASE_IN_DIR,  f"{task}_metadata.jsonl")
        out_file = os.path.join(BASE_OUT_DIR, f"{task}_triplets.jsonl")
        process_task(task, in_file, out_file, num_pos_per_anchor=k)
        if os.path.exists(out_file):
            with open(out_file, 'r') as f:
                cnt = sum(1 for _ in f)
                total_triplets += cnt

    print("-" * 50)
    print(f"🎉 Done! Total generated triplets across all tasks: {total_triplets}")
