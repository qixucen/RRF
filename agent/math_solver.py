"""数学答案提取工具"""

import re
from api.llm import gen

MATH_PROMPT_SUFFIX = "\n\nSolve this step by step. Put your final answer in \\boxed{}."


def extract_boxed_answer(text: str, fallback_last_number: bool = True) -> str | None:
    """
    从 LLM 输出中提取 \\boxed{} 格式的答案
    
    支持嵌套括号，如 \\boxed{3\\sqrt{13}} 或 \\boxed{\\frac{1}{2}}
    
    Args:
        text: LLM 的输出文本
        fallback_last_number: 如果没有 boxed，是否回退到提取最后一个数字
    
    Returns:
        提取的答案字符串，失败返回 None
    """
    if not text:
        return None
    
    # 优先提取 \boxed{...} 中的内容（支持一层嵌套括号）
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # fallback: 提取最后一个数字
    if fallback_last_number:
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if nums:
            return nums[-1]
    
    return None


def extract_numeric_answer(text: str, fallback_last_number: bool = True) -> float | int | None:
    """
    从 LLM 输出中提取数值答案
    
    Args:
        text: LLM 的输出文本
        fallback_last_number: 如果没有 boxed，是否回退到提取最后一个数字
    
    Returns:
        提取的数值（int 或 float），失败返回 None
    """
    answer = extract_boxed_answer(text, fallback_last_number)
    if answer is None:
        return None
    
    # 清理答案，只保留数字和小数点
    cleaned = re.sub(r'[^\d.\-]', '', answer)
    if not cleaned or cleaned in ['.', '-', '-.']:
        return None
    
    try:
        if '.' in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return None


async def solve_math(problem: str, model: str = "gpt-4o-mini", temperature: float = 0.7, 
                     add_prompt_suffix: bool = True) -> tuple[str | None, any]:
    """
    解决数学问题并提取答案
    
    Args:
        problem: 数学问题描述
        model: 使用的模型
        temperature: 温度参数
        add_prompt_suffix: 是否自动添加 boxed 后缀
    
    Returns:
        (提取的答案, 完整响应) 元组
    """
    prompt = problem + MATH_PROMPT_SUFFIX if add_prompt_suffix else problem
    response, _ = await gen(prompt=prompt, model=model, temperature=temperature)
    answer = extract_boxed_answer(response) if response else None
    return answer, response
