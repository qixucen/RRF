"""收集 AIME24 数据集的多次采样结果
用于分析 limit scaling（每题收集 1024 次响应）

Usage:
    python collect_limit.py --runs 100  # 每次运行收集 100 条（会自动续写）
    python collect_limit.py --runs 50 --problems 5  # 只跑前 5 题，每题 50 条
    python collect_limit.py --status  # 查看当前进度
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.math_solver import solve_math, extract_boxed_answer

# 配置
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TARGET_RUNS = 2000  # 目标收集次数
DATA_PATH = "dataset/data/aime24.json"


def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))


def get_limit_dir() -> str:
    """获取 limit 输出目录"""
    return os.path.join(get_script_dir(), "log", "limit")


def get_results_file(problem_id: str) -> str:
    """获取问题的结果文件路径（单个 JSON 文件）"""
    return os.path.join(get_limit_dir(), f"{problem_id}.json")


def load_existing_results(problem_id: str) -> list[dict]:
    """加载已有的采样结果"""
    results_file = get_results_file(problem_id)
    if not os.path.exists(results_file):
        return []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(problem_id: str, results: list[dict]):
    """保存采样结果"""
    results_file = get_results_file(problem_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def count_existing_runs(problem_id: str) -> int:
    """统计已有的运行次数"""
    return len(load_existing_results(problem_id))


def load_problems(data_path: str) -> list[dict]:
    """加载数据集"""
    if not os.path.isabs(data_path):
        data_path = os.path.join(get_script_dir(), data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_status(problems: list[dict], target_runs: int) -> dict:
    """获取所有问题的收集状态"""
    status = {}
    for problem in problems:
        pid = problem["id"]
        existing = count_existing_runs(pid)
        status[pid] = {
            "existing": existing,
            "target": target_runs,
            "remaining": max(0, target_runs - existing),
            "progress": f"{existing}/{target_runs} ({existing/target_runs*100:.1f}%)"
        }
    return status


def print_status(problems: list[dict], target_runs: int):
    """打印收集状态"""
    status = get_all_status(problems, target_runs)
    
    print("=" * 70)
    print(f"收集进度 (目标: 每题 {target_runs} 次)")
    print("=" * 70)
    
    total_existing = 0
    total_target = len(problems) * target_runs
    
    for pid, s in status.items():
        total_existing += s["existing"]
        bar_len = 30
        filled = int(bar_len * s["existing"] / target_runs)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"{pid:15s} [{bar}] {s['progress']}")
    
    print("=" * 70)
    print(f"总计: {total_existing}/{total_target} ({total_existing/total_target*100:.1f}%)")
    print("=" * 70)


async def run_single(problem: dict, model: str, temperature: float) -> dict:
    """运行单次采样"""
    try:
        answer, response = await solve_math(
            problem=problem["problem"],
            model=model,
            temperature=temperature
        )
        
        # 提取答案
        extracted = answer or ""
        if not extracted and response:
            extracted = extract_boxed_answer(response, fallback_last_number=True) or ""
        
        # 判断正确性
        correct = str(extracted).strip() == str(problem["answer"]).strip()
        
        return {
            "response": response or "",
            "extracted": extracted,
            "correct": correct
        }
        
    except Exception as e:
        return {
            "response": f"ERROR: {str(e)}",
            "extracted": "",
            "correct": False
        }


async def collect_for_problem(
    problem: dict,
    num_runs: int,
    model: str,
    temperature: float,
    target_runs: int,
    pbar: tqdm
) -> int:
    """为单个问题收集数据"""
    pid = problem["id"]
    
    # 加载已有结果
    existing_results = load_existing_results(pid)
    existing_count = len(existing_results)
    remaining_to_target = max(0, target_runs - existing_count)
    
    # 本次要运行的数量（不超过目标剩余量）
    runs_this_time = min(num_runs, remaining_to_target)
    
    if runs_this_time <= 0:
        pbar.update(num_runs)  # 更新进度条（跳过）
        return 0
    
    # 并发运行
    tasks = [run_single(problem, model, temperature) for _ in range(runs_this_time)]
    
    # 收集新结果
    new_results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        new_results.append(result)
        pbar.update(1)
    
    # 合并并保存
    all_results = existing_results + new_results
    save_results(pid, all_results)
    
    # 补足进度条更新（如果 num_runs > runs_this_time）
    if num_runs > runs_this_time:
        pbar.update(num_runs - runs_this_time)
    
    return len(new_results)


def parse_id_list(id_str: str) -> list[str]:
    """解析 ID 列表，支持 '1,2,3' 或 'aime24_1,aime24_2' 格式"""
    if not id_str:
        return []
    
    ids = []
    for part in id_str.split(","):
        part = part.strip()
        if part.isdigit():
            # 数字格式，自动补全前缀
            ids.append(f"aime24_{part}")
        else:
            ids.append(part)
    return ids


async def main():
    parser = argparse.ArgumentParser(
        description="收集 AIME24 数据集的多次采样结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python collect_limit.py --runs 100           # 每题收集 100 次（续写）
    python collect_limit.py --runs 50 -p 5       # 只跑前 5 题
    python collect_limit.py --exclude 10         # 排除第 10 题
    python collect_limit.py --exclude 10,8       # 排除第 10 和 8 题
    python collect_limit.py --only 1,2,3         # 只跑第 1,2,3 题
    python collect_limit.py --status             # 查看进度
"""
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=100,
        help="本次每题要收集的数量 (默认: 100)"
    )
    
    parser.add_argument(
        "--problems", "-p",
        type=int,
        default=None,
        help="只处理前 N 题 (默认: 全部 30 题)"
    )
    
    parser.add_argument(
        "--exclude", "-e",
        type=str,
        default=None,
        help="排除的题目 ID，逗号分隔 (如: 10 或 10,8 或 aime24_10)"
    )
    
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只运行的题目 ID，逗号分隔 (如: 1,2,3 或 aime24_1,aime24_2)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"使用的模型 (默认: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"温度参数 (默认: {DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="只显示当前收集状态，不运行"
    )
    
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_RUNS,
        help=f"每题目标收集次数 (默认: {DEFAULT_TARGET_RUNS})"
    )
    
    args = parser.parse_args()
    target_runs = args.target
    
    # 加载数据
    problems = load_problems(DATA_PATH)
    
    # 限制题目数量
    if args.problems:
        problems = problems[:args.problems]
    
    # 只选中特定题目
    if args.only:
        only_ids = set(parse_id_list(args.only))
        problems = [p for p in problems if p["id"] in only_ids]
    
    # 排除特定题目
    if args.exclude:
        exclude_ids = set(parse_id_list(args.exclude))
        problems = [p for p in problems if p["id"] not in exclude_ids]
    
    print(f"数据集: {DATA_PATH}")
    print(f"题目数量: {len(problems)}")
    print(f"目标次数: {target_runs}/题")
    
    # 只显示状态
    if args.status:
        print_status(problems, target_runs)
        return
    
    # 显示当前状态
    print("\n当前进度:")
    print_status(problems, target_runs)
    
    print(f"\n本次运行配置:")
    print(f"  模型: {args.model}")
    print(f"  温度: {args.temperature}")
    print(f"  每题收集: {args.runs} 次")
    print("=" * 70)
    
    # 计算实际需要运行的总数
    total_to_run = 0
    for problem in problems:
        existing = count_existing_runs(problem["id"])
        remaining = max(0, target_runs - existing)
        total_to_run += min(args.runs, remaining)
    
    if total_to_run == 0:
        print("\n所有题目都已达到目标次数！")
        return
    
    print(f"\n开始收集 (本次最多 {total_to_run} 条)...")
    start_time = datetime.now()
    
    # 进度条：以实际需要运行的数量为总数
    total_collected = 0
    
    with tqdm(total=len(problems) * args.runs, desc="总进度", unit="次") as pbar:
        for problem in problems:
            collected = await collect_for_problem(
                problem=problem,
                num_runs=args.runs,
                model=args.model,
                temperature=args.temperature,
                target_runs=target_runs,
                pbar=pbar
            )
            total_collected += collected
    
    elapsed = datetime.now() - start_time
    print(f"\n完成！本次收集 {total_collected} 条，耗时 {elapsed}")
    
    # 显示最终状态
    print("\n最终进度:")
    print_status(problems, target_runs)


if __name__ == "__main__":
    asyncio.run(main())
