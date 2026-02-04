"""Baseline runner for comparing different agent types
Usage:
    python run_baselines.py --agent recap --dataset math500 --n 30
    python run_baselines.py --agent recursive --dataset math500 --n 30 --max-depth 2
    python run_baselines.py --agent none --dataset math500 --n 30
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

# Global log file
JSONL_FILE = None

from agent.recursive_agent import recursive_gen
from agent.baselines.recap_agent import recap_gen
from agent.math_solver import solve_math, extract_boxed_answer
from dataset import load_local_dataset, list_datasets
from math_verify import parse, verify


def verify_answer(extracted: str, expected: str, dataset_name: str) -> bool:
    """Verify answer based on dataset type"""
    if not extracted or not expected:
        return False
    
    if dataset_name in ("aime24", "aime25"):
        return extracted.strip() == expected.strip()
    
    try:
        def wrap_latex(s: str) -> str:
            s = s.strip()
            if not (s.startswith('$') or s.startswith('\\boxed')):
                return f'${s}$'
            return s
        
        gold = parse(wrap_latex(expected))
        pred = parse(wrap_latex(extracted))
        
        if gold and pred:
            return verify(gold, pred)
        return extracted.strip() == expected.strip()
    except Exception:
        return extracted.strip() == expected.strip()


async def run_with_agent(
    agent_type: str,
    prompt: str,
    model: str,
    max_iterations: int,
    max_depth: int = 3,
    force_invoke: bool = False,
    ctx_window: int = 10,
    verbose: bool = False
) -> tuple[str, dict]:
    """Run a prompt with the specified agent type"""
    
    if agent_type == "none":
        answer, response = await solve_math(prompt, model=model)
        meta = {
            "total_calls": 1,
            "iterations": [{"response": response}],
            "stopped_reason": "return"
        }
        return answer or response, meta
    
    elif agent_type == "recursive":
        return await recursive_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth,
            max_iterations=max_iterations,
            verbose=verbose,
            force_invoke=force_invoke
        )
    
    elif agent_type == "recap":
        return await recap_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth,
            max_iterations=max_iterations,
            verbose=verbose,
            ctx_window=ctx_window
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def run_single(problem: dict, agent_type: str, config: dict, dataset_name: str) -> dict:
    """Run a single problem"""
    global JSONL_FILE
    try:
        content, meta = await run_with_agent(
            agent_type=agent_type,
            prompt=problem["problem"],
            **config
        )
        
        extracted = extract_boxed_answer(content or "") or ""
        correct = verify_answer(extracted, problem["answer"], dataset_name)
        
        result = {
            "id": problem["id"],
            "answer": problem["answer"],
            "extracted": extracted,
            "correct": correct,
            "total_calls": meta.get("total_calls", 0),
            "stopped_reason": meta.get("stopped_reason", ""),
            "depth_used": meta.get("depth_used", 0),
        }
        
        # Write to jsonl
        if JSONL_FILE:
            log_entry = {
                "problem_id": problem["id"],
                "agent_type": agent_type,
                "config": {k: v for k, v in config.items() if k != "verbose"},
                "total_calls": meta.get("total_calls", 0),
                "stopped_reason": meta.get("stopped_reason", ""),
                "depth_used": meta.get("depth_used", 0),
                "final_answer": content,
                "extracted": extracted,
                "correct": correct,
                "iterations": meta.get("iterations", [])
            }
            JSONL_FILE.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            JSONL_FILE.flush()
        
        return result
    except Exception as e:
        result = {
            "id": problem["id"],
            "answer": problem["answer"],
            "extracted": "",
            "correct": False,
            "error": str(e)
        }
        # Log error too
        if JSONL_FILE:
            log_entry = {
                "problem_id": problem["id"],
                "agent_type": agent_type,
                "error": str(e)
            }
            JSONL_FILE.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            JSONL_FILE.flush()
        return result


async def run_benchmark(
    agent_type: str,
    problems: list,
    config: dict,
    dataset_name: str,
    num_runs: int = 1
) -> list:
    """Run benchmark on problems"""
    tasks = []
    for _ in range(num_runs):
        for problem in problems:
            tasks.append(run_single(problem, agent_type, config, dataset_name))
    
    results = []
    with tqdm(total=len(tasks), desc=agent_type, unit="problem") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    return results


def analyze(results: list) -> dict:
    """Analyze results"""
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    avg_calls = sum(r.get("total_calls", 0) for r in results) / total if total else 0
    avg_depth = sum(r.get("depth_used", 0) for r in results) / total if total else 0
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "avg_calls": avg_calls,
        "avg_depth": avg_depth
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline runner for agent comparison")
    
    parser.add_argument("--agent", "-a", type=str, 
                        choices=["none", "recursive", "recap"],
                        default="recap", help="Agent type (default: recap)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--dataset", "-D", type=str, choices=list_datasets(), default="math500")
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--num", "-n", type=int, default=None, help="Number of problems")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Runs per problem")
    
    # Agent config
    parser.add_argument("--max-depth", "-d", type=int, default=2)
    parser.add_argument("--max-iterations", "-i", type=int, default=10)
    parser.add_argument("--ctx-window", "-c", type=int, default=10, help="[recap] Context window")
    parser.add_argument("--force-invoke", action="store_true", help="[recursive] Force invoke")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


async def main():
    global JSONL_FILE
    args = parse_args()
    
    # Calculate range
    start, end = args.start, args.end
    if args.num:
        end = start + args.num
    
    # Load data
    problems = load_local_dataset(args.dataset)[start:end]
    
    # Build config
    config = {
        "model": args.model,
        "max_iterations": args.max_iterations,
        "max_depth": args.max_depth,
        "verbose": args.verbose,
    }
    if args.agent == "recursive":
        config["force_invoke"] = args.force_invoke
    elif args.agent == "recap":
        config["ctx_window"] = args.ctx_window
    
    # Setup log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    log_name = f"{args.agent}_{timestamp}"
    jsonl_path = os.path.join(log_dir, f"{log_name}.jsonl")
    JSONL_FILE = open(jsonl_path, 'w', encoding='utf-8')
    
    # Print info
    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Dataset: {args.dataset} [{start}:{end}] ({len(problems)} problems)")
    print(f"Config: {config}")
    print(f"Runs: {args.runs}")
    print(f"Log: {jsonl_path}")
    print("=" * 60)
    
    try:
        # Run
        results = await run_benchmark(
            agent_type=args.agent,
            problems=problems,
            config=config,
            dataset_name=args.dataset,
            num_runs=args.runs
        )
        
        # Analyze
        analysis = analyze(results)
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Accuracy: {analysis['correct']}/{analysis['total']} = {analysis['accuracy']*100:.1f}%")
        print(f"Avg calls: {analysis['avg_calls']:.1f}")
        if analysis['avg_depth'] > 0:
            print(f"Avg depth: {analysis['avg_depth']:.1f}")
        
        # Save results
        results_path = os.path.join(log_dir, f"{log_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "config": {
                    "agent_type": args.agent,
                    "model": args.model,
                    "max_depth": args.max_depth,
                    "max_iterations": args.max_iterations,
                    "ctx_window": config.get("ctx_window"),
                    "dataset": args.dataset,
                    "start": start,
                    "end": end,
                    "num_runs": args.runs
                },
                "analysis": analysis,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Detailed log: {jsonl_path}")
    finally:
        if JSONL_FILE:
            JSONL_FILE.close()
            JSONL_FILE = None


if __name__ == "__main__":
    asyncio.run(main())
