"""Sequential Recursive Agent - Sequential exploration with recursive reasoning

Multiple explorations run sequentially, each new exploration can see the complete
results of all previous explorations, allowing it to try different approaches
or correct previous errors.

Finally, use LLM to intelligently aggregate all exploration results instead of simple voting.
"""

import asyncio
from agent.base_agent import BaseAgent
from agent.compressed_recursive_agent import CompressedRecursiveAgent
from api.llm import gen


class SequentialRecursiveAgent:
    """
    Sequential exploration recursive Agent coordinator.
    
    Key features:
    1. Launch multiple explorations sequentially (CompressedRecursiveAgent)
    2. Each new exploration can see complete results of all previous explorations
    3. New explorations can try different approaches or correct previous errors
    4. Finally use LLM to intelligently aggregate all results (not simple voting)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_depth: int = 3,
        max_iterations: int = 10,
        num_explorations: int = 3,
        verbose: bool = True,
        force_invoke: bool = False,
    ):
        self.model = model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.num_explorations = num_explorations
        self.verbose = verbose
        self.force_invoke = force_invoke
        
        # Statistics
        self.total_calls = 0
        self._world_state_calls = 0
        self._aggregation_calls = 0
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[Sequential] {message}")
    
    def _build_exploration_prompt(self, original_prompt: str, previous_results: list[dict]) -> str:
        """
        Build prompt with previous exploration results.
        
        Allow new exploration to:
        1. See previous answers and reasoning
        2. Try different approaches
        3. Correct previous errors
        """
        if not previous_results:
            # First exploration, return original problem directly
            return original_prompt
        
        # Build summary of previous explorations
        previous_summaries = []
        for i, r in enumerate(previous_results):
            content = r.get("content", "")[:800]  # Truncate
            previous_summaries.append(f"""
=== Exploration {i + 1} Result ===
{content}
""")
        
        enhanced_prompt = f"""{original_prompt}

---
Previous exploration attempts (for reference, you can use different approaches or correct possible errors):
{"".join(previous_summaries)}
---

Based on the above information, provide your solution. If previous explorations have errors, point them out and correct them.
If you believe the previous answer is correct, you can verify using a similar approach.
The final answer must be wrapped in \\boxed{{}}.
"""
        return enhanced_prompt
    
    async def _run_single_exploration(
        self,
        original_prompt: str,
        exploration_id: int,
        previous_results: list[dict]
    ) -> dict:
        """Run a single exploration"""
        self._log(f"Starting exploration {exploration_id + 1}/{self.num_explorations}")
        
        # Build enhanced prompt (including previous exploration results)
        enhanced_prompt = self._build_exploration_prompt(original_prompt, previous_results)
        
        agent = CompressedRecursiveAgent(
            model=self.model,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            force_invoke=self.force_invoke,
        )
        
        result = await agent.run(enhanced_prompt)
        
        return {
            "exploration_id": exploration_id,
            "content": result.get("content", ""),
            "total_calls": agent.total_calls,
            "world_state_calls": agent._world_state_calls,
            "stopped_reason": result.get("stopped_reason", ""),
            "iterations": result.get("iterations", []),
            "deepest_depth_used": result.get("deepest_depth_used", 0)
        }
    
    async def _aggregate_explorations(self, prompt: str, results: list[dict]) -> dict:
        """
        Intelligently aggregate all exploration results.
        
        Not simple voting, but use LLM to analyze all answers and reasoning processes,
        synthesize to get the most reliable answer.
        """
        self._log("Aggregating exploration results...")
        self._aggregation_calls += 1
        
        # Build exploration result summaries
        exploration_summaries = []
        for r in results:
            content = r.get("content", "")[:1000]  # Truncate
            exploration_summaries.append(f"""
Exploration {r['exploration_id'] + 1}:
- Answer: {content}
- LLM calls: {r.get('total_calls', 0)}
- Stopped reason: {r.get('stopped_reason', '')}
""")
        
        aggregation_prompt = f"""You are an answer aggregation expert. Multiple sequential explorations attempted to solve the same problem, where later explorations can see results from earlier ones. Now you need to synthesize their results.

Original problem:
{prompt}

Results from each exploration (in order):
{"".join(exploration_summaries)}

Please comprehensively analyze all exploration results and provide the most reliable final answer. Requirements:
1. Analyze the evolution of answers across explorations (did later explorations correct earlier errors)
2. If answers are consistent, adopt directly
3. If answers differ, analyze which is more likely correct (consider completeness of reasoning and corrections)
4. Final answer must be wrapped in \\boxed{{}}

Output format:
Brief analysis (1-2 sentences), then provide the final answer."""
        
        response, _ = await gen(
            prompt=aggregation_prompt,
            model=self.model,
            temperature=0.3,
            max_tokens=500
        )
        
        return {
            "content": response or "",
            "exploration_results": results
        }
    
    async def run(self, prompt: str) -> dict:
        """
        Run sequential exploration.
        
        1. Launch explorations sequentially, each can see previous results
        2. Intelligently aggregate results
        """
        self._log(f"Starting {self.num_explorations} sequential explorations...")
        
        results = []
        
        # Execute explorations sequentially
        for i in range(self.num_explorations):
            result = await self._run_single_exploration(prompt, i, results)
            results.append(result)
            
            # Count calls
            self.total_calls += result.get("total_calls", 0)
            self._world_state_calls += result.get("world_state_calls", 0)
            
            self._log(f"Exploration {i + 1} completed. Answer: {result.get('content', '')[:100]}...")
        
        # Aggregate results
        aggregated = await self._aggregate_explorations(prompt, results)
        self.total_calls += 1  # Aggregation call
        
        self._log(f"Completed. Total LLM calls: {self.total_calls}")
        
        return {
            "content": aggregated["content"],
            "exploration_results": results,
            "total_calls": self.total_calls,
            "world_state_calls": self._world_state_calls,
            "aggregation_calls": self._aggregation_calls,
            "stopped_reason": "aggregated"
        }


async def sequential_recursive_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_iterations: int = 10,
    num_explorations: int = 3,
    verbose: bool = True,
    force_invoke: bool = False
) -> tuple[str, dict]:
    """
    Sequential exploration recursive reasoning.
    
    Args:
        prompt: The problem to solve
        model: LLM model to use
        max_depth: Maximum depth for subtask decomposition
        max_iterations: Maximum iterations per agent
        num_explorations: Number of sequential explorations
        verbose: Whether to print progress
        force_invoke: If True, must invoke enough subtasks to unlock return.
    
    Returns:
        (content, meta) tuple
    """
    agent = SequentialRecursiveAgent(
        model=model,
        max_depth=max_depth,
        max_iterations=max_iterations,
        num_explorations=num_explorations,
        verbose=verbose,
        force_invoke=force_invoke
    )
    
    result = await agent.run(prompt)
    
    meta = {
        "total_calls": agent.total_calls,
        "world_state_calls": agent._world_state_calls,
        "aggregation_calls": agent._aggregation_calls,
        "num_explorations": num_explorations,
        "max_depth": max_depth,
        "force_invoke": force_invoke,
        "exploration_results": result.get("exploration_results", []),
        "stopped_reason": result["stopped_reason"]
    }
    
    return result["content"], meta


async def test():
    """Test sequential exploration"""
    prompt = """What is 123 * 456 + 789?"""
    
    print("=" * 60)
    print("Sequential Recursive Agent Test")
    print(f"Prompt: {prompt}")
    print(f"Settings: num_explorations=3, max_depth=2")
    print("=" * 60)
    
    content, meta = await sequential_recursive_gen(
        prompt=prompt,
        model="gpt-4o-mini",
        max_depth=2,
        max_iterations=10,
        num_explorations=3,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print(f"Final Aggregated Answer: {content}")
    print(f"\nMeta:")
    print(f"  - Total LLM calls: {meta['total_calls']}")
    print(f"  - World state calls: {meta['world_state_calls']}")
    print(f"  - Aggregation calls: {meta['aggregation_calls']}")
    print(f"  - Explorations: {meta['num_explorations']}")
    print("=" * 60)
    
    # Print individual exploration results
    print("\nIndividual exploration results:")
    for r in meta.get("exploration_results", []):
        from agent.math_solver import extract_boxed_answer
        answer = extract_boxed_answer(r.get("content", ""))
        print(f"  E{r['exploration_id'] + 1}: {answer} (calls: {r['total_calls']})")


if __name__ == "__main__":
    asyncio.run(test())
