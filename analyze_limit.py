"""统计 log/limit 目录下每道题的正确率

Usage:
    python analyze_limit.py
"""

import json
import os
from glob import glob

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def analyze():
    limit_dir = os.path.join(get_script_dir(), "log", "limit")
    
    # 获取所有 JSON 文件
    json_files = sorted(glob(os.path.join(limit_dir, "*.json")))
    
    if not json_files:
        print("没有找到数据文件")
        return
    
    results = []
    total_correct = 0
    total_samples = 0
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        problem_id = filename.replace(".json", "")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        n_samples = len(data)
        n_correct = sum(1 for item in data if item.get("correct", False))
        accuracy = n_correct / n_samples if n_samples > 0 else 0
        
        results.append({
            "id": problem_id,
            "samples": n_samples,
            "correct": n_correct,
            "accuracy": accuracy
        })
        
        total_correct += n_correct
        total_samples += n_samples
    
    # 按 id 数字排序
    def sort_key(x):
        # 从 aime24_1 提取数字 1
        parts = x["id"].split("_")
        return int(parts[-1]) if parts[-1].isdigit() else 0
    
    results.sort(key=sort_key)
    
    # 打印结果
    print("=" * 60)
    print(f"{'问题ID':<15} {'样本数':>8} {'正确数':>8} {'正确率':>10}")
    print("=" * 60)
    
    for r in results:
        bar_len = 20
        filled = int(bar_len * r["accuracy"])
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"{r['id']:<15} {r['samples']:>8} {r['correct']:>8} {r['accuracy']*100:>9.1f}% [{bar}]")
    
    print("=" * 60)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"{'总计':<15} {total_samples:>8} {total_correct:>8} {overall_accuracy*100:>9.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
