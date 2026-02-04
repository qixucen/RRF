"""打印指定题目的所有正确结果，输出到临时文件

Usage:
    python show_correct.py 26           # 打印 aime24_26 的正确结果
    python show_correct.py aime24_26    # 同上
    python show_correct.py 1 --limit 5  # 只打印前 5 条
"""

import argparse
import json
import os
import tempfile


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_results_file(problem_id: str) -> str:
    """获取问题的结果文件路径"""
    # 支持直接输入数字
    if problem_id.isdigit():
        problem_id = f"aime24_{problem_id}"
    return os.path.join(get_script_dir(), "log", "limit", f"{problem_id}.json")


def load_problem(problem_id: str) -> dict | None:
    """从数据集加载原始问题"""
    data_path = os.path.join(get_script_dir(), "dataset", "data", "aime24.json")
    if not os.path.exists(data_path):
        return None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    for p in problems:
        if p["id"] == problem_id:
            return p
    return None


def show_correct(problem_id: str, limit: int = None):
    """打印指定题目的正确结果到临时文件"""
    # 标准化 id
    if problem_id.isdigit():
        problem_id = f"aime24_{problem_id}"
    
    filepath = get_results_file(problem_id)
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct_results = [item for item in data if item.get('correct', False)]
    
    # 加载原始问题
    original_problem = load_problem(problem_id)
    
    # 创建临时文件
    tmp_dir = os.path.join(get_script_dir(), "log", "temp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{problem_id}_correct.txt")
    
    with open(tmp_file, 'w', encoding='utf-8') as out:
        out.write(f"题目: {problem_id}\n")
        out.write(f"共 {len(correct_results)} 条正确结果（总共 {len(data)} 条，正确率 {len(correct_results)/len(data)*100:.1f}%）\n")
        out.write("\n")
        
        # 写入原始问题
        if original_problem:
            out.write(f"{'='*60}\n")
            out.write("原问题:\n")
            out.write(f"{'='*60}\n")
            out.write(original_problem["problem"])
            out.write(f"\n\n正确答案: {original_problem['answer']}\n")
            out.write("\n")
        
        if not correct_results:
            out.write("没有正确结果\n")
        else:
            # 限制输出数量
            results_to_show = correct_results[:limit] if limit else correct_results
            
            for i, item in enumerate(results_to_show, 1):
                out.write(f"{'='*60}\n")
                out.write(f"正确结果 {i}\n")
                out.write(f"提取答案: {item['extracted']}\n")
                out.write(f"{'='*60}\n")
                out.write(item['response'])
                out.write("\n\n")
    
    print(f"题目: {problem_id}")
    print(f"共 {len(correct_results)} 条正确结果（总共 {len(data)} 条，正确率 {len(correct_results)/len(data)*100:.1f}%）")
    print(f"结果已保存到: {tmp_file}")


def main():
    parser = argparse.ArgumentParser(
        description="打印指定题目的所有正确结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python show_correct.py 26           # 打印 aime24_26 的正确结果
    python show_correct.py aime24_26    # 同上
    python show_correct.py 1 --limit 5  # 只打印前 5 条
"""
    )
    
    parser.add_argument(
        "id",
        type=str,
        help="题目 ID (如: 26 或 aime24_26)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="限制输出数量"
    )
    
    args = parser.parse_args()
    show_correct(args.id, args.limit)


if __name__ == "__main__":
    main()
