"""Parallel Recursive Agent - Parallel exploration with recursive reasoning

Multiple explorations run in parallel, each exploration records results to shared state
after invoke completion. Other explorations can see these completed discoveries when
preparing context for child tasks.

Finally, use LLM to intelligently aggregate all exploration results instead of simple voting.
"""

import asyncio
import time
from agent.base_agent import BaseAgent
from agent.compressed_recursive_agent import CompressedRecursiveAgent
from api.llm import gen


class SharedExplorationState:
    """
    Shared exploration state, allowing parallel explorations to see
    completed invoke results from other explorations.
    
    Uses invoke completion as the recording point, not every LLM call.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._discoveries: list[dict] = []  # All completed invoke results
    
    async def record_discovery(
        self, 
        exploration_id: int, 
        depth: int,
        task: str, 
        result: str,
        compressed_result: str = None
    ):
        """Record a completed invoke (discovery)"""
        async with self._lock:
            self._discoveries.append({
                "exploration_id": exploration_id,
                "depth": depth,
                "task": task[:200],  # Truncate
                "result": result[:500] if result else "",
                "compressed_result": compressed_result[:300] if compressed_result else "",
                "timestamp": time.time()
            })
    
    async def get_others_discoveries(self, my_exploration_id: int) -> list[dict]:
        """Get all completed discoveries from other explorations"""
        async with self._lock:
            return [d for d in self._discoveries 
                    if d["exploration_id"] != my_exploration_id]
    
    async def get_all_discoveries(self) -> list[dict]:
        """Get all discoveries (for debugging/logging)"""
        async with self._lock:
            return list(self._discoveries)
    
    async def get_stats(self) -> dict:
        """Get statistics"""
        async with self._lock:
            by_exploration = {}
            for d in self._discoveries:
                eid = d["exploration_id"]
                by_exploration[eid] = by_exploration.get(eid, 0) + 1
            return {
                "total_discoveries": len(self._discoveries),
                "by_exploration": by_exploration
            }


class ParallelExplorationAgent(CompressedRecursiveAgent):
    """
    Single parallel exploration Agent, inherits from CompressedRecursiveAgent.
    
    Improvements:
    1. After invoke completion, record results to shared state
    2. When preparing context for child, can view discoveries from other explorations
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_depth: int = 3,
        max_iterations: int = 10,
        verbose: bool = True,
        force_invoke: bool = False,
        _current_depth: int = None,
        _parent_context: str = None,
        # Parallel exploration specific parameters
        exploration_id: int = 0,
        shared_state: SharedExplorationState = None
    ):
        super().__init__(
            model=model,
            max_depth=max_depth,
            max_iterations=max_iterations,
            verbose=verbose,
            force_invoke=force_invoke,
            _current_depth=_current_depth,
            _parent_context=_parent_context
        )
        self.exploration_id = exploration_id
        self._shared_state = shared_state
    
    def _log(self, message: str, indent: int = 0):
        if self.verbose:
            prefix = "  " * (self.max_depth - self._current_depth)
            print(f"{prefix}[P{self.exploration_id}|D{self._current_depth}] {message}")
    
    async def _prepare_context_for_child(self, task: str, current_messages: list) -> str:
        """
        Override: When preparing child task context, also reference completed
        discoveries from other explorations.
        """
        self._log("ParallelWorldState: Preparing context for child...")
        self._world_state_calls += 1
        
        # Build current conversation summary
        conversation_summary = []
        for msg in current_messages[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:500]
            if role in ("user", "assistant"):
                conversation_summary.append(f"[{role}]: {content}")
        
        # Get completed discoveries from other explorations
        other_discoveries = []
        if self._shared_state:
            other_discoveries = await self._shared_state.get_others_discoveries(self.exploration_id)
        
        # Build summary of other explorations' discoveries
        other_discoveries_text = ""
        if other_discoveries:
            # Only look at recent discoveries to avoid context overflow
            recent_discoveries = sorted(other_discoveries, key=lambda x: x["timestamp"], reverse=True)[:5]
            discovery_items = []
            for d in recent_discoveries:
                summary = d.get("compressed_result") or d.get("result", "")
                discovery_items.append(f"- Exploration{d['exploration_id']}(depth{d['depth']}): {summary[:100]}")
            other_discoveries_text = f"""

Recent discoveries from other parallel explorations (for reference, avoid duplicate exploration):
{chr(10).join(discovery_items)}
"""
        
        prompt = f"""You are a context management assistant. The main task is decomposing subtasks, and you need to decide what information to pass to the subtask.

Current conversation history summary:
{chr(10).join(conversation_summary)}
{other_discoveries_text}
Subtask description:
{task}

Please extract the most helpful context information for this subtask. Requirements:
1. Only extract information directly related to the subtask
2. If other explorations have relevant discoveries, reference them but avoid completely duplicate exploration directions
3. Be concise, no more than 150 words
4. If no context needs to be passed, output "None"

Output format: Directly output the refined context without extra formatting."""
        
        response, _ = await gen(
            prompt=prompt,
            model=self.model,
            temperature=0.3,
            max_tokens=300
        )
        
        context = (response or "").strip()
        if context and context.lower() != "none":
            self._log(f"ParallelWorldState: Context prepared: {context[:50]}...")
            return context
        return None
    
    async def execute_tool(self, name: str, args: dict) -> str | dict:
        """Override execute_tool: Record to shared state after invoke completion"""
        task = args.get("task", "")
        
        if name == "invoke":
            self._log(f"invoke: {task[:50]}...")
            self._invoke_count += 1
            
            if self._current_depth > 0:
                # === World State Maintenance Phase 1: Prepare child task context ===
                child_context = await self._prepare_context_for_child(
                    task, 
                    getattr(self, '_current_messages', [])
                )
                
                # Create child agent (maintain parallel exploration identity)
                sub_agent = ParallelExplorationAgent(
                    model=self.model,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations,
                    verbose=self.verbose,
                    force_invoke=self.force_invoke,
                    _current_depth=self._current_depth - 1,
                    _parent_context=child_context,
                    exploration_id=self.exploration_id,
                    shared_state=self._shared_state
                )
                
                result = await sub_agent.run(task)
                
                # Count calls
                self.total_calls += sub_agent.total_calls
                self._world_state_calls += sub_agent._world_state_calls
                
                # Track deepest level
                sub_depth = 1 + sub_agent.deepest_depth_used
                self.deepest_depth_used = max(self.deepest_depth_used, sub_depth)
                
                # === World State Maintenance Phase 2: Compress child task result ===
                raw_content = result.get("content", "")
                if raw_content:
                    compressed_content = await self._compress_child_result(task, raw_content)
                else:
                    compressed_content = None
                
                # === Parallel exploration specific: Record discovery to shared state ===
                if self._shared_state:
                    await self._shared_state.record_discovery(
                        exploration_id=self.exploration_id,
                        depth=self._current_depth - 1,
                        task=task,
                        result=raw_content,
                        compressed_result=compressed_content
                    )
                    self._log(f"ParallelWorldState: Discovery recorded to shared state")
                
                # Build return result
                if self.force_invoke:
                    remaining = self._current_depth - self._invoke_count
                    if remaining > 0:
                        unlock_hint = f"[Invoke {remaining} more subtask(s) to unlock return]"
                    else:
                        unlock_hint = "[return is now unlocked.]"
                else:
                    unlock_hint = ""
                
                if compressed_content:
                    result_str = f"Subtask result: {compressed_content}\n\n{unlock_hint}"
                else:
                    result_str = f"Subtask failed: {result['stopped_reason']}\n\n{unlock_hint}"
                
                return {
                    "result": result_str,
                    "subtask_info": {
                        "depth": self._current_depth - 1,
                        "total_calls": sub_agent.total_calls,
                        "world_state_calls": sub_agent._world_state_calls,
                        "stopped_reason": result["stopped_reason"],
                        "raw_content": raw_content,
                        "compressed_content": compressed_content,
                        "parent_context": child_context,
                        "iterations": result.get("iterations", []),
                        "deepest_depth_used": sub_agent.deepest_depth_used
                    }
                }
            else:
                # Deepest level: Direct LLM call
                self._log(f"invoke (direct): {task[:50]}...")
                
                if self._parent_context:
                    enhanced_task = f"Context: {self._parent_context}\n\nTask: {task}"
                else:
                    enhanced_task = task
                
                response, _ = await gen(prompt=enhanced_task, model=self.model, temperature=0.7)
                llm_response = response or "No response"
                
                # Record discovery at deepest level too
                if self._shared_state:
                    await self._shared_state.record_discovery(
                        exploration_id=self.exploration_id,
                        depth=0,
                        task=task,
                        result=llm_response,
                        compressed_result=None
                    )
                
                return {
                    "result": f"Subtask result: {llm_response}",
                    "llm_call": {
                        "task": task,
                        "response": llm_response
                    }
                }
        
        return f"Unknown tool: {name}"


class ParallelRecursiveAgent:
    """
    Parallel exploration recursive Agent coordinator.
    
    Key features:
    1. Launch multiple explorations in parallel (ParallelExplorationAgent)
    2. All explorations share a SharedExplorationState
    3. After each exploration's invoke completion, results are immediately visible to others
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
            print(f"[Parallel] {message}")
    
    async def _run_single_exploration(
        self,
        prompt: str,
        exploration_id: int,
        shared_state: SharedExplorationState
    ) -> dict:
        """Run a single exploration"""
        self._log(f"Starting exploration {exploration_id}")
        
        agent = ParallelExplorationAgent(
            model=self.model,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            force_invoke=self.force_invoke,
            exploration_id=exploration_id,
            shared_state=shared_state
        )
        
        result = await agent.run(prompt)
        
        self._log(f"Exploration {exploration_id} completed")
        
        return {
            "exploration_id": exploration_id,
            "content": result.get("content", ""),
            "total_calls": agent.total_calls,
            "world_state_calls": agent._world_state_calls,
            "stopped_reason": result.get("stopped_reason", ""),
            "iterations": result.get("iterations", []),
            "deepest_depth_used": result.get("deepest_depth_used", 0)
        }
    
    async def _aggregate_explorations(self, prompt: str, results: list[dict], shared_state: SharedExplorationState) -> dict:
        """
        Intelligently aggregate all exploration results.
        """
        self._log("Aggregating exploration results...")
        self._aggregation_calls += 1
        
        # Get shared state statistics
        stats = await shared_state.get_stats()
        
        # Build exploration result summaries
        exploration_summaries = []
        for r in results:
            content = r.get("content", "")[:1000]
            exploration_summaries.append(f"""
Exploration {r['exploration_id']}:
- Answer: {content}
- LLM calls: {r.get('total_calls', 0)}
- Stopped reason: {r.get('stopped_reason', '')}
""")
        
        aggregation_prompt = f"""You are an answer aggregation expert. Multiple parallel explorations simultaneously attempted to solve the same problem, and they could see each other's intermediate discoveries. Now you need to synthesize their results.

Original problem:
{prompt}

Shared state statistics: Total {stats['total_discoveries']} intermediate discoveries generated

Final results from each exploration:
{"".join(exploration_summaries)}

Please comprehensively analyze all exploration results and provide the most reliable final answer. Requirements:
1. Analyze consistency and differences among exploration answers
2. If answers are consistent, adopt directly
3. If answers differ, analyze which is more likely correct (consider completeness of reasoning)
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
        Run parallel exploration.
        
        1. Create shared state
        2. Launch all explorations in parallel
        3. Wait for all to complete
        4. Intelligently aggregate results
        """
        self._log(f"Starting {self.num_explorations} parallel explorations...")
        
        # Create shared state
        shared_state = SharedExplorationState()
        
        # Launch all explorations in parallel
        tasks = [
            self._run_single_exploration(prompt, i, shared_state)
            for i in range(self.num_explorations)
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Count calls
        for r in results:
            self.total_calls += r.get("total_calls", 0)
            self._world_state_calls += r.get("world_state_calls", 0)
        
        # Get shared state statistics
        stats = await shared_state.get_stats()
        self._log(f"Shared state: {stats['total_discoveries']} discoveries recorded")
        
        # Aggregate results
        aggregated = await self._aggregate_explorations(prompt, results, shared_state)
        self.total_calls += 1  # Aggregation call
        
        self._log(f"Completed. Total LLM calls: {self.total_calls}")
        
        return {
            "content": aggregated["content"],
            "exploration_results": results,
            "total_calls": self.total_calls,
            "world_state_calls": self._world_state_calls,
            "aggregation_calls": self._aggregation_calls,
            "shared_discoveries": stats["total_discoveries"],
            "stopped_reason": "aggregated"
        }


async def parallel_recursive_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_iterations: int = 10,
    num_explorations: int = 3,
    verbose: bool = True,
    force_invoke: bool = False
) -> tuple[str, dict]:
    """
    Parallel exploration recursive reasoning.
    
    Args:
        prompt: The problem to solve
        model: LLM model to use
        max_depth: Maximum depth for subtask decomposition
        max_iterations: Maximum iterations per agent
        num_explorations: Number of parallel explorations
        verbose: Whether to print progress
        force_invoke: If True, must invoke enough subtasks to unlock return.
    
    Returns:
        (content, meta) tuple
    """
    agent = ParallelRecursiveAgent(
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
        "shared_discoveries": result.get("shared_discoveries", 0),
        "exploration_results": result.get("exploration_results", []),
        "stopped_reason": result["stopped_reason"]
    }
    
    return result["content"], meta


async def test():
    """Test parallel exploration"""
    prompt = """What is 123 * 456 + 789?"""
    
    print("=" * 60)
    print("Parallel Recursive Agent Test")
    print(f"Prompt: {prompt}")
    print(f"Settings: num_explorations=3, max_depth=2")
    print("=" * 60)
    
    content, meta = await parallel_recursive_gen(
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
    print(f"  - Shared discoveries: {meta['shared_discoveries']}")
    print(f"  - Explorations: {meta['num_explorations']}")
    print("=" * 60)
    
    # Print individual exploration results
    print("\nIndividual exploration results:")
    for r in meta.get("exploration_results", []):
        from agent.math_solver import extract_boxed_answer
        answer = extract_boxed_answer(r.get("content", ""))
        print(f"  P{r['exploration_id']}: {answer} (calls: {r['total_calls']})")


if __name__ == "__main__":
    asyncio.run(test())
