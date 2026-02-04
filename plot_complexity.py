"""分析复杂度与准确率的关系

- none agent: 分析响应长度与正确率的关系
- chain/recursive/variant1: 分析调用次数与正确率的关系

Usage:
    python plot_complexity.py log/none_*.json
    python plot_complexity.py log/chain_*.json --bins 5
    python plot_complexity.py log/*.json --compare
"""

import argparse
import json
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> tuple[list[dict], dict]:
    """加载结果文件（新格式）
    
    新格式:
    {
        "config": {...},
        "analysis": {...},
        "results": [  # 列表，不是字典
            {"id": ..., "correct": ..., "total_calls": ..., "final_content": ..., ...},
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)
    
    config = data.get("config", {})
    results = data.get("results", [])
    
    # results 应该是列表
    if isinstance(results, list):
        return results, config
    
    # 兼容旧格式（results 是字典）
    if isinstance(results, dict):
        flat_results = []
        for config_key, items in results.items():
            for item in items:
                item["config_key"] = config_key
                flat_results.append(item)
        return flat_results, config
    
    return [], config


def get_response_length(result: dict) -> int:
    """获取响应长度（用于 none agent）
    
    使用 iterations[0].agent_response 的长度，这是 LLM 的完整响应
    """
    # 从 iterations 中获取完整响应长度
    iterations = result.get("iterations", [])
    if iterations:
        first_iter = iterations[0]
        response = first_iter.get("agent_response", "")
        if response:
            return len(response)
    
    # 降级: 使用 final_content
    if "final_content" in result and result["final_content"]:
        return len(result["final_content"])
    
    return 0


def analyze_none_agent(results: list[dict], bins: int = 5) -> dict:
    """分析 none agent: 响应长度 vs 准确率"""
    raw_data = []
    for r in results:
        length = get_response_length(r)
        correct = 1 if r.get("correct", False) else 0
        raw_data.append((length, correct))
    
    if not raw_data:
        return {"raw_data": [], "binned_data": [], "stats": {}}
    
    lengths = [x[0] for x in raw_data]
    correct_list = [x[1] for x in raw_data]
    
    # 分桶统计
    min_len, max_len = min(lengths), max(lengths)
    
    if bins > 0 and max_len > min_len:
        bin_edges = np.linspace(min_len, max_len + 1, bins + 1)
        binned_data = []
        
        for i in range(bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_items = [(l, c) for l, c in raw_data if low <= l < high]
            
            if bin_items:
                bin_center = (low + high) / 2
                bin_correct = sum(x[1] for x in bin_items)
                bin_total = len(bin_items)
                bin_acc = bin_correct / bin_total
                binned_data.append((bin_center, bin_acc, bin_total))
    else:
        # 按实际长度值分组（可能太细，一般用分桶）
        by_len = defaultdict(list)
        for l, c in raw_data:
            by_len[l].append(c)
        
        binned_data = []
        for length, corrs in sorted(by_len.items()):
            acc = sum(corrs) / len(corrs)
            binned_data.append((length, acc, len(corrs)))
    
    # 总体统计
    total_correct = sum(correct_list)
    total = len(correct_list)
    avg_len = np.mean(lengths)
    
    stats = {
        "total": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0,
        "avg_length": avg_len,
        "min_length": min_len,
        "max_length": max_len
    }
    
    return {
        "raw_data": raw_data,
        "binned_data": binned_data,
        "stats": stats,
        "x_label": "Response Length",
        "metric": "length"
    }


def analyze_agent_calls(results: list[dict], bins: int = 5) -> dict:
    """分析 agent 调用次数 vs 准确率（chain/recursive/variant1）"""
    raw_data = []
    for r in results:
        calls = r.get("total_calls", 1)
        correct = 1 if r.get("correct", False) else 0
        raw_data.append((calls, correct))
    
    if not raw_data:
        return {"raw_data": [], "binned_data": [], "stats": {}}
    
    calls_list = [x[0] for x in raw_data]
    correct_list = [x[1] for x in raw_data]
    
    # 分桶统计
    min_calls, max_calls = min(calls_list), max(calls_list)
    
    if bins > 0 and max_calls > min_calls:
        bin_edges = np.linspace(min_calls, max_calls + 1, bins + 1)
        binned_data = []
        
        for i in range(bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_items = [(c, corr) for c, corr in raw_data if low <= c < high]
            
            if bin_items:
                bin_center = (low + high) / 2
                bin_correct = sum(x[1] for x in bin_items)
                bin_total = len(bin_items)
                bin_acc = bin_correct / bin_total
                binned_data.append((bin_center, bin_acc, bin_total))
    else:
        # 按实际 calls 值分组
        by_calls = defaultdict(list)
        for c, corr in raw_data:
            by_calls[c].append(corr)
        
        binned_data = []
        for calls, corrs in sorted(by_calls.items()):
            acc = sum(corrs) / len(corrs)
            binned_data.append((calls, acc, len(corrs)))
    
    # 总体统计
    total_correct = sum(correct_list)
    total = len(correct_list)
    avg_calls = np.mean(calls_list)
    
    stats = {
        "total": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0,
        "avg_calls": avg_calls,
        "min_calls": min_calls,
        "max_calls": max_calls
    }
    
    return {
        "raw_data": raw_data,
        "binned_data": binned_data,
        "stats": stats,
        "x_label": "Total Calls",
        "metric": "calls"
    }


def analyze_results(results: list[dict], agent_type: str, bins: int = 5) -> dict:
    """根据 agent 类型选择分析方法"""
    if agent_type == "none":
        return analyze_none_agent(results, bins)
    else:
        return analyze_agent_calls(results, bins)


def plot_single(results: list[dict], config: dict, output_path: str = None, bins: int = 5):
    """绘制单个实验的复杂度-准确率图"""
    agent_type = config.get("agent_type", "unknown")
    analysis = analyze_results(results, agent_type, bins=bins)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_label = analysis.get("x_label", "Complexity")
    metric = analysis.get("metric", "calls")
    
    # 图1: 散点图 + 回归线
    ax1 = axes[0]
    raw_data = analysis["raw_data"]
    x_vals = [x[0] for x in raw_data]
    correct = [x[1] for x in raw_data]
    
    # 添加抖动以便看清重叠点
    jitter = np.random.uniform(-0.1, 0.1, len(correct))
    ax1.scatter(x_vals, [c + j for c, j in zip(correct, jitter)], 
                alpha=0.5, c=['green' if c else 'red' for c in correct], s=50)
    
    # 添加回归线
    if len(set(x_vals)) > 1:
        z = np.polyfit(x_vals, correct, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        ax1.plot(x_line, p(x_line), "b--", alpha=0.8, label=f"Trend: y={z[0]:.6f}x+{z[1]:.2f}")
        ax1.legend()
    
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Correct (0/1)", fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Wrong", "Correct"])
    ax1.set_title(f"Scatter: {x_label} vs Correctness", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 分桶准确率柱状图
    ax2 = axes[1]
    binned_data = analysis["binned_data"]
    
    if binned_data:
        centers = [x[0] for x in binned_data]
        accs = [x[1] for x in binned_data]
        counts = [x[2] for x in binned_data]
        
        bars = ax2.bar(range(len(centers)), accs, color='steelblue', alpha=0.7, edgecolor='black')
        
        # 在柱子上显示样本数
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xticks(range(len(centers)))
        # 格式化标签
        if metric == "length":
            ax2.set_xticklabels([f'{c:.0f}' for c in centers], rotation=45)
        else:
            ax2.set_xticklabels([f'{c:.1f}' for c in centers])
        ax2.set_xlabel(f"{x_label} (binned)", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0, 1.15)
        ax2.set_title(f"Accuracy by {x_label} Bin", fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加总体统计
    stats = analysis["stats"]
    dataset = config.get("dataset", "unknown")
    title = f"{agent_type.upper()} on {dataset}"
    
    if metric == "length":
        subtitle = f"(Total: {stats['total']}, Acc: {stats['accuracy']*100:.1f}%, Avg Length: {stats['avg_length']:.0f})"
    else:
        subtitle = f"(Total: {stats['total']}, Acc: {stats['accuracy']*100:.1f}%, Avg Calls: {stats['avg_calls']:.1f})"
    
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return analysis


def plot_compare(file_paths: list[str], output_path: str = None):
    """对比多个实验的复杂度-准确率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10.colors
    
    # 分开 none 和其他 agent
    none_data = []
    agent_data = []
    
    for path in file_paths:
        results, config = load_results(path)
        agent_type = config.get("agent_type", "unknown")
        
        if agent_type == "none":
            none_data.append((path, results, config))
        else:
            agent_data.append((path, results, config))
    
    # 左图: none agent (length vs accuracy)
    ax1 = axes[0]
    for i, (path, results, config) in enumerate(none_data):
        analysis = analyze_none_agent(results, bins=0)
        binned_data = analysis["binned_data"]
        
        if not binned_data:
            continue
        
        lengths = [x[0] for x in binned_data]
        accs = [x[1] for x in binned_data]
        counts = [x[2] for x in binned_data]
        
        # 按长度排序计算累积准确率
        cum_correct = 0
        cum_total = 0
        cum_accs = []
        cum_lens = []
        
        sorted_data = sorted(zip(lengths, accs, counts))
        for l, acc, n in sorted_data:
            cum_correct += acc * n
            cum_total += n
            cum_accs.append(cum_correct / cum_total)
            cum_lens.append(l)
        
        dataset = config.get("dataset", os.path.basename(path))
        label = f"none ({dataset}, n={analysis['stats']['total']})"
        
        ax1.plot(cum_lens, cum_accs, marker='o', color=colors[i % len(colors)], 
                label=label, linewidth=2, markersize=4, alpha=0.8)
    
    ax1.set_xlabel("Response Length", fontsize=12)
    ax1.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax1.set_title("NONE Agent: Length vs Accuracy", fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 右图: agent (calls vs accuracy)
    ax2 = axes[1]
    for i, (path, results, config) in enumerate(agent_data):
        agent_type = config.get("agent_type", "unknown")
        analysis = analyze_agent_calls(results, bins=0)
        binned_data = analysis["binned_data"]
        
        if not binned_data:
            continue
        
        calls = [x[0] for x in binned_data]
        accs = [x[1] for x in binned_data]
        counts = [x[2] for x in binned_data]
        
        # 按 calls 排序计算累积准确率
        cum_correct = 0
        cum_total = 0
        cum_accs = []
        cum_calls = []
        
        sorted_data = sorted(zip(calls, accs, counts))
        for c, acc, n in sorted_data:
            cum_correct += acc * n
            cum_total += n
            cum_accs.append(cum_correct / cum_total)
            cum_calls.append(c)
        
        dataset = config.get("dataset", "")
        depth = config.get("max_depth")
        label = f"{agent_type}"
        if depth:
            label += f" (d={depth})"
        label += f" (n={analysis['stats']['total']})"
        
        ax2.plot(cum_calls, cum_accs, marker='o', color=colors[i % len(colors)], 
                label=label, linewidth=2, markersize=6)
    
    ax2.set_xlabel("Total Calls", fontsize=12)
    ax2.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax2.set_title("Agents: Calls vs Accuracy", fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析复杂度与准确率的关系")
    parser.add_argument("files", nargs="+", help="结果文件路径 (支持 glob 模式)")
    parser.add_argument("--bins", type=int, default=5, help="分桶数量 (default: 5)")
    parser.add_argument("--compare", action="store_true", help="对比模式")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出图片路径")
    
    args = parser.parse_args()
    
    # 展开 glob 模式
    all_files = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        if matched:
            all_files.extend(matched)
        elif os.path.exists(pattern):
            all_files.append(pattern)
    
    if not all_files:
        print("No files found!")
        return
    
    print(f"Found {len(all_files)} file(s)")
    
    if args.compare and len(all_files) > 1:
        plot_compare(all_files, args.output)
    else:
        for path in all_files:
            print(f"\nAnalyzing: {path}")
            results, config = load_results(path)
            agent_type = config.get("agent_type", "unknown")
            
            output = args.output
            if not output:
                output = path.replace(".json", "_complexity.png")
            
            analysis = plot_single(results, config, output, bins=args.bins)
            
            # 打印统计
            stats = analysis["stats"]
            metric = analysis.get("metric", "calls")
            
            print(f"  Agent: {agent_type}")
            print(f"  Total: {stats['total']}")
            print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
            
            if metric == "length":
                print(f"  Avg Length: {stats['avg_length']:.0f}")
                print(f"  Length Range: {stats['min_length']} - {stats['max_length']}")
            else:
                print(f"  Avg Calls: {stats['avg_calls']:.1f}")
                print(f"  Calls Range: {stats['min_calls']} - {stats['max_calls']}")


if __name__ == "__main__":
    main()
