"""
统一数据集下载工具

支持的数据集:
- AIME 2024 (美国邀请赛数学竞赛)
- AIME 2025
- MATH-500 (500道数学题目基准测试)
"""

import json
import os
from pathlib import Path
from typing import Literal

from datasets import load_dataset


# ============================================================================
# 配置
# ============================================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 数据集 Hugging Face 源配置
DATASET_SOURCES = {
    "aime24": {
        "hf_path": "math-ai/aime24",
        "description": "AIME 2024 - 30道数学竞赛题目",
    },
    "aime25": {
        "hf_path": "math-ai/aime25",
        "description": "AIME 2025 - 30道数学竞赛题目",
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "description": "MATH-500 - 500道数学题目基准测试",
    },
}

# 默认保存路径
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

DatasetName = Literal["aime24", "aime25", "math500"]


# ============================================================================
# 数据下载函数
# ============================================================================

def download_dataset(
    name: DatasetName,
    save_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    下载指定数据集
    
    Args:
        name: 数据集名称 ("aime24", "aime25" 或 "math500")
        save_dir: 保存目录，默认为 dataset/data/
        force: 是否强制重新下载（覆盖已有文件）
    
    Returns:
        保存的文件路径
    
    Raises:
        ValueError: 不支持的数据集名称
    """
    if name not in DATASET_SOURCES:
        raise ValueError(f"不支持的数据集: {name}，可选: {list(DATASET_SOURCES.keys())}")
    
    source = DATASET_SOURCES[name]
    save_dir = Path(save_dir) if save_dir else DEFAULT_DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"{name}.json"
    
    # 检查是否已存在
    if save_path.exists() and not force:
        print(f"[{name}] 数据集已存在: {save_path}")
        return save_path
    
    print(f"[{name}] 正在从 Hugging Face 下载: {source['hf_path']}")
    
    # 下载数据集
    ds = load_dataset(source["hf_path"], split="test")
    
    # 转换为统一格式
    data = _normalize_dataset(name, ds)
    
    # 保存为 JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[{name}] 已保存 {len(data)} 条数据到: {save_path}")
    return save_path


def _extract_boxed_answer(solution: str) -> str:
    """
    从 \\boxed{xxx} 格式中提取答案（支持嵌套括号）
    
    Args:
        solution: 包含 \\boxed{} 的字符串
    
    Returns:
        提取的答案，如果无法提取则返回原字符串
    """
    import re
    # 匹配 \boxed{...} 格式（支持一层嵌套括号）
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    if match:
        return match.group(1)
    return solution


def _normalize_dataset(name: str, ds) -> list[dict]:
    """
    将不同来源的数据集标准化为统一格式
    
    统一格式:
    {
        "id": str,           # 题目ID
        "problem": str,      # 题目描述
        "answer": str,       # 标准答案
        "source": str,       # 来源数据集
        "year": int,         # 年份
    }
    """
    data = []
    
    if name == "aime24":
        # aime24 数据集答案字段是 "solution"，格式为 \boxed{xxx}
        for i, item in enumerate(ds):
            solution = item.get("solution", item.get("answer", ""))
            answer = _extract_boxed_answer(str(solution))
            data.append({
                "id": f"aime24_{i+1}",
                "problem": item.get("problem", item.get("question", "")),
                "answer": answer,
                "source": "aime24",
                "year": 2024,
            })
    
    elif name == "aime25":
        for i, item in enumerate(ds):
            data.append({
                "id": item.get("id", f"aime25_{i+1}"),
                "problem": item.get("problem", item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "source": "aime25",
                "year": 2025,
            })
    
    elif name == "math500":
        # MATH-500 数据集字段: problem, answer, subject, level, unique_id
        for i, item in enumerate(ds):
            answer = item.get("answer", "")
            # 如果答案是 \boxed{} 格式，提取内容
            if isinstance(answer, str) and "\\boxed" in answer:
                answer = _extract_boxed_answer(answer)
            data.append({
                "id": item.get("unique_id", f"math500_{i+1}"),
                "problem": item.get("problem", ""),
                "answer": str(answer),
                "source": "math500",
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
            })
    
    return data


def download_all(save_dir: str | Path | None = None, force: bool = False) -> dict[str, Path]:
    """
    下载所有支持的数据集
    
    Args:
        save_dir: 保存目录
        force: 是否强制重新下载
    
    Returns:
        数据集名称到保存路径的映射
    """
    paths = {}
    for name in DATASET_SOURCES:
        try:
            paths[name] = download_dataset(name, save_dir, force)
        except Exception as e:
            print(f"[{name}] 下载失败: {e}")
    return paths


# ============================================================================
# 数据加载函数
# ============================================================================

def load_local_dataset(name: DatasetName, data_dir: str | Path | None = None) -> list[dict]:
    """
    加载本地数据集（如不存在则自动下载）
    
    Args:
        name: 数据集名称
        data_dir: 数据目录
    
    Returns:
        数据列表
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_path = data_dir / f"{name}.json"
    
    # 如果本地不存在，先下载
    if not data_path.exists():
        download_dataset(name, data_dir)
    
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aime24(data_dir: str | Path | None = None) -> list[dict]:
    """加载 AIME 2024 数据集"""
    return load_local_dataset("aime24", data_dir)


def load_aime25(data_dir: str | Path | None = None) -> list[dict]:
    """加载 AIME 2025 数据集"""
    return load_local_dataset("aime25", data_dir)


def load_math500(data_dir: str | Path | None = None) -> list[dict]:
    """加载 MATH-500 数据集"""
    return load_local_dataset("math500", data_dir)


def get_dataset_info(name: DatasetName) -> dict:
    """获取数据集信息"""
    if name not in DATASET_SOURCES:
        raise ValueError(f"不支持的数据集: {name}")
    return DATASET_SOURCES[name].copy()


def list_datasets() -> list[str]:
    """列出所有支持的数据集"""
    return list(DATASET_SOURCES.keys())


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIME 数据集下载工具")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["aime24", "aime25", "math500", "all"],
        default="all",
        help="要下载的数据集 (默认: all)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="保存目录 (默认: dataset/data/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有支持的数据集"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("支持的数据集:")
        for name, info in DATASET_SOURCES.items():
            print(f"  - {name}: {info['description']}")
        return
    
    if args.dataset == "all":
        download_all(args.save_dir, args.force)
    else:
        download_dataset(args.dataset, args.save_dir, args.force)


if __name__ == "__main__":
    main()
