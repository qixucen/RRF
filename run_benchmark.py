"""Benchmark runner for different agent types
Usage:
    python run_benchmark.py --agent chain --dataset aime24 --n 100
    python run_benchmark.py --agent recursive --dataset math500 --max-depth 3 --n 100
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from tqdm import tqdm

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.chain_agent import chain_gen
from agent.recursive_agent import recursive_gen
from agent.compressed_recursive_agent import compressed_recursive_gen
from agent.sequential_recursive_agent import sequential_recursive_gen
from agent.parallel_recursive_agent import parallel_recursive_gen
from agent.math_solver import solve_math, extract_boxed_answer
from dataset import load_local_dataset, list_datasets
from math_verify import parse, verify

# Global results state for real-time updates
RESULTS_PATH = None
RESULTS_CONFIG = None
RESULTS_DATA = []  # [results]
RESULTS_LOCK = asyncio.Lock()


async def save_results_realtime(result: dict):
    """Save results to .json in real-time"""
    global RESULTS_PATH, RESULTS_CONFIG, RESULTS_DATA
    async with RESULTS_LOCK:
        if not RESULTS_PATH:
            return
        
        # Add result to data
        RESULTS_DATA.append(result)
        
        # Compute analysis
        analysis = analyze_results(RESULTS_DATA)
        
        # Write to file
        with open(RESULTS_PATH, 'w') as f:
            json.dump({
                "config": RESULTS_CONFIG,
                "analysis": analysis,
                "results": RESULTS_DATA
            }, f, indent=2, ensure_ascii=False)


def verify_answer(extracted: str, expected: str, dataset_name: str) -> bool:
    """Verify answer based on dataset type
    
    - AIME datasets: simple string comparison (answers are integers)
    - MATH500: use math_verify for LaTeX expression comparison
    """
    if not extracted or not expected:
        return False
    
    # AIME answers are integers, use simple comparison
    if dataset_name in ("aime24", "aime25"):
        return extracted.strip() == expected.strip()
    
    # MATH500: use math_verify for complex expressions
    try:
        # Wrap in $...$ for proper parsing if not already formatted
        def wrap_latex(s: str) -> str:
            s = s.strip()
            if not (s.startswith('$') or s.startswith('\\boxed')):
                return f'${s}$'
            return s
        
        gold = parse(wrap_latex(expected))
        pred = parse(wrap_latex(extracted))
        
        if gold and pred:
            return verify(gold, pred)
        # Fallback to string comparison if parsing fails
        return extracted.strip() == expected.strip()
    except Exception:
        # Fallback to string comparison if parsing fails
        return extracted.strip() == expected.strip()


async def run_with_agent(
    agent_type: str,
    prompt: str,
    model: str,
    max_iterations: int,
    max_depth: int = None,
    force_invoke: bool = True,
    num_explorations: int = None,
    verbose: bool = False
) -> tuple[str, dict]:
    """Run a prompt with the specified agent type"""
    
    if agent_type == "none":
        # Simple single LLM call (baseline, no agent)
        answer, response = await solve_math(prompt, model=model)
        meta = {
            "total_calls": 1,
            "tool_calls": [],
            "iterations": [{
                "iteration": 1,
                "agent_response": response,
                "action": "return",
                "args": {"answer": answer or ""},
                "result": answer or ""
            }],
            "stopped_reason": "return"
        }
        return answer or response, meta
    
    elif agent_type == "chain":
        return await chain_gen(
            prompt=prompt,
            model=model,
            max_iterations=max_iterations,
            verbose=verbose
        )
    elif agent_type == "recursive":
        return await recursive_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth or 3,
            max_iterations=max_iterations,
            verbose=verbose,
            force_invoke=force_invoke
        )
    elif agent_type == "compressed_recursive":
        return await compressed_recursive_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth or 3,
            max_iterations=max_iterations,
            verbose=verbose,
            force_invoke=force_invoke
        )
    elif agent_type == "sequential_recursive":
        return await sequential_recursive_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth or 3,
            max_iterations=max_iterations,
            num_explorations=num_explorations or 3,
            verbose=verbose,
            force_invoke=force_invoke
        )
    elif agent_type == "parallel_recursive":
        return await parallel_recursive_gen(
            prompt=prompt,
            model=model,
            max_depth=max_depth or 3,
            max_iterations=max_iterations,
            num_explorations=num_explorations or 3,
            verbose=verbose,
            force_invoke=force_invoke
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def run_single_problem(
    problem: dict,
    agent_type: str,
    model: str,
    max_iterations: int,
    max_depth: int,
    force_invoke: bool,
    num_explorations: int,
    dataset_name: str,
    verbose: bool
) -> dict:
    """Run a single problem and return result"""
    try:
        content, meta = await run_with_agent(
            agent_type=agent_type,
            prompt=problem["problem"],
            model=model,
            max_iterations=max_iterations,
            max_depth=max_depth,
            force_invoke=force_invoke,
            num_explorations=num_explorations,
            verbose=verbose
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
            "final_content": content or "",
            "iterations": meta.get("iterations", [])
        }
        if "depth_used" in meta:
            result["depth_used"] = meta.get("depth_used", 0)
        if "world_state_calls" in meta:
            result["world_state_calls"] = meta.get("world_state_calls", 0)
        if "aggregation_calls" in meta:
            result["aggregation_calls"] = meta.get("aggregation_calls", 0)
        if "num_explorations" in meta:
            result["num_explorations"] = meta.get("num_explorations", 0)
        if "shared_discoveries" in meta:
            result["shared_discoveries"] = meta.get("shared_discoveries", 0)
        
        return result
        
    except Exception as e:
        return {
            "id": problem["id"],
            "answer": problem["answer"],
            "extracted": "",
            "correct": False,
            "error": str(e),
            "iterations": []
        }


async def run_benchmark(
    dataset_name: str,
    agent_type: str,
    model: str,
    max_iterations: int,
    max_depth: int,
    force_invoke: bool,
    num_explorations: int,
    num_runs: int,
    start: int,
    end: int | None,
    verbose: bool
) -> list:
    """Run benchmark"""
    
    # Load data
    problems = load_local_dataset(dataset_name)
    
    # Slice problems
    problems = problems[start:end]
    
    print(f"Loaded {len(problems)} problems")
    print(f"Agent: {agent_type}")
    if agent_type in ("recursive", "compressed_recursive", "sequential_recursive", "parallel_recursive"):
        print(f"Max depth: {max_depth}")
        print(f"Force invoke: {force_invoke}")
    if agent_type in ("sequential_recursive", "parallel_recursive"):
        print(f"Num explorations: {num_explorations}")
    print(f"Max iterations: {max_iterations}")
    print(f"Runs per config: {num_runs}")
    print(f"Model: {model}")
    print("=" * 60)
    
    # Build task list
    tasks = []
    for _ in range(num_runs):
        for problem in problems:
            tasks.append(run_single_problem(
                problem, agent_type, model, max_iterations, max_depth, force_invoke, num_explorations, dataset_name, verbose
            ))
    
    # Run with progress bar
    results = []
    with tqdm(total=len(tasks), desc=agent_type, unit="problem") as pbar:
        # Run tasks concurrently and update progress as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Save results in real-time
            await save_results_realtime(result)
            pbar.update(1)
    
    return results


def analyze_results(results: list) -> dict:
    """Analyze benchmark results"""
    total = len(results)
    if total == 0:
        return {"correct": 0, "total": 0, "accuracy": 0, "avg_calls": 0}
    
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / total
    avg_calls = sum(r.get("total_calls", 0) for r in results) / total
    
    analysis = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "avg_calls": avg_calls,
    }
    
    # Add depth stats if available
    if any("depth_used" in r for r in results):
        avg_depth = sum(r.get("depth_used", 0) for r in results) / total
        analysis["avg_depth_used"] = avg_depth
    
    # Add world state calls stats if available (compressed_recursive, spatial_recursive)
    if any("world_state_calls" in r for r in results):
        avg_ws_calls = sum(r.get("world_state_calls", 0) for r in results) / total
        analysis["avg_world_state_calls"] = avg_ws_calls
    
    # Add aggregation calls stats if available (sequential_recursive, parallel_recursive)
    if any("aggregation_calls" in r for r in results):
        avg_agg_calls = sum(r.get("aggregation_calls", 0) for r in results) / total
        analysis["avg_aggregation_calls"] = avg_agg_calls
    
    # Add shared discoveries stats if available (parallel_recursive)
    if any("shared_discoveries" in r for r in results):
        avg_shared = sum(r.get("shared_discoveries", 0) for r in results) / total
        analysis["avg_shared_discoveries"] = avg_shared
    
    return analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Agent types:
  none                 - Single LLM call with boxed answer prompt (baseline, no agent)
  chain                - Uses call_llm tool to query LLM for subtasks (no recursion)
  recursive            - Uses invoke tool to decompose into self-contained subtasks (with depth control)
  compressed_recursive - Recursive with world state maintenance (context refinement + result compression)
  sequential_recursive - Multiple sequential explorations, each seeing previous results, with intelligent aggregation
  parallel_recursive   - Multiple parallel explorations with real-time shared discoveries, with intelligent aggregation

Examples:
  python run_benchmark.py --agent none --dataset aime24 --runs 5
  python run_benchmark.py --agent chain --dataset math500 --start 0 --num 50
  python run_benchmark.py --agent recursive --dataset math500 --start 100 --end 200
  python run_benchmark.py --agent recursive --dataset math500 --n 30 --max-depth 3
  python run_benchmark.py --agent compressed_recursive --dataset aime24 --max-depth 2
  python run_benchmark.py --agent sequential_recursive --dataset math500 --num-explorations 3 --max-depth 2
  python run_benchmark.py --agent parallel_recursive --dataset math500 --num-explorations 3 --max-depth 2

For depth comparison experiments, use compare_depths.py instead.
"""
    )
    
    # Agent selection
    parser.add_argument(
        "--agent", "-a",
        type=str,
        choices=["none", "chain", "recursive", "compressed_recursive", "sequential_recursive", "parallel_recursive"],
        default="none",
        help="Agent type: none (single call), chain (call_llm), recursive (invoke), compressed_recursive (recursive+world_state), sequential_recursive (sequential explorations), parallel_recursive (parallel explorations) (default: none)"
    )
    
    # Agent config
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=3,
        help="[recursive/compressed_recursive] Max depth for subtask decomposition (default: 3)"
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=10,
        help="Max iterations per agent (default: 10)"
    )
    parser.add_argument(
        "--force-invoke",
        action="store_true",
        help="[recursive/compressed_recursive/sequential_recursive/parallel_recursive] Must invoke enough subtasks before return is unlocked (default: allow free choice)"
    )
    parser.add_argument(
        "--num-explorations", "-E",
        type=int,
        default=3,
        help="[sequential_recursive/parallel_recursive] Number of explorations (default: 3)"
    )
    
    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    
    # Data
    available_datasets = list_datasets()
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        choices=available_datasets,
        default="math500",
        help=f"Dataset name: {', '.join(available_datasets)} (default: math500)"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start index of problems (0-based, default: 0)"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End index of problems (exclusive, default: all)"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of problems to use (alternative to --end, use with --start)"
    )
    
    # Runs
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of runs per problem (default: 1)"
    )
    
    # Verbose
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_name = args.dataset
    
    # Calculate range
    start = args.start
    end = args.end
    if args.num is not None:
        end = start + args.num
    
    # Format range string for display
    range_str = f"[{start}:{end}]" if end else f"[{start}:]"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print(f"Benchmark - {args.agent.upper()} Agent")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_name} {range_str}")
    print("=" * 60)
    
    # Setup log directory and results file
    log_dir = os.path.join(script_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    log_name = f"{args.agent}_{timestamp}"
    results_path = os.path.join(log_dir, f"{log_name}.json")
    
    global RESULTS_PATH, RESULTS_CONFIG, RESULTS_DATA
    
    # Initialize real-time results
    RESULTS_PATH = results_path
    RESULTS_CONFIG = {
        "agent_type": args.agent,
        "model": args.model,
        "max_iterations": args.max_iterations,
        "max_depth": args.max_depth if args.agent in ("recursive", "compressed_recursive", "sequential_recursive", "parallel_recursive") else None,
        "force_invoke": args.force_invoke if args.agent in ("recursive", "compressed_recursive", "sequential_recursive", "parallel_recursive") else None,
        "num_explorations": args.num_explorations if args.agent in ("sequential_recursive", "parallel_recursive") else None,
        "num_runs": args.runs,
        "dataset": dataset_name,
        "start": start,
        "end": end
    }
    RESULTS_DATA = []
    
    print(f"Results (real-time): {results_path}")
    
    # Run benchmark
    results = await run_benchmark(
        dataset_name=dataset_name,
        agent_type=args.agent,
        model=args.model,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        force_invoke=args.force_invoke,
        num_explorations=args.num_explorations,
        num_runs=args.runs,
        start=start,
        end=end,
        verbose=args.verbose
    )
    
    # Analyze
    analysis = analyze_results(results)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy: {analysis['correct']}/{analysis['total']} = {analysis['accuracy']*100:.1f}%")
    print(f"  Avg LLM calls: {analysis['avg_calls']:.1f}")
    if "avg_depth_used" in analysis:
        print(f"  Avg depth used: {analysis['avg_depth_used']:.1f}")
    if "avg_world_state_calls" in analysis:
        print(f"  Avg world state calls: {analysis['avg_world_state_calls']:.1f}")
    if "avg_aggregation_calls" in analysis:
        print(f"  Avg aggregation calls: {analysis['avg_aggregation_calls']:.1f}")
    if "avg_shared_discoveries" in analysis:
        print(f"  Avg shared discoveries: {analysis['avg_shared_discoveries']:.1f}")
    
    # Save final results (also saved in real-time during run)
    with open(results_path, 'w') as f:
        json.dump({
            "config": RESULTS_CONFIG,
            "analysis": analysis,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
