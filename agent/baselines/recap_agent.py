"""ReCAP Agent - Recursive Context-Aware Reasoning and Planning
Based on: https://arxiv.org/abs/2510.23822

Aligned with official implementation:
- Single agent loop with node pointer movement
- Context truncation by deleting middle history
- Structured re-injection with (T, S[1:])
"""

import asyncio
import json
from typing import List, Optional
from api.llm import gen


class Node:
    """Node in the task decomposition tree (aligned with ReCAP)"""
    
    def __init__(self, task_name: str, parent: Optional['Node'] = None):
        self.task_name = task_name
        self.parent = parent
        self.children: List['Node'] = []
        self.info_list: List[dict] = []  # List of {think, subtasks}
    
    def add_child(self, child: 'Node'):
        self.children.append(child)
        child.parent = self
    
    def set_info(self, info: dict):
        """Store response info"""
        self.info_list.append(info)
    
    def get_latest_info(self) -> Optional[dict]:
        """Get latest {think, subtasks}"""
        return self.info_list[-1] if self.info_list else None


class ReCapAgent:
    """
    ReCAP Agent - Single loop with pointer movement
    
    Aligned with official implementation:
    - node_ptr moves through task tree
    - during_down flag for descent/ascent
    - Context truncation by deleting middle
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_depth: int = 3,
        max_iterations: int = 50,
        verbose: bool = True,
        ctx_window: int = 16,  # Max history length before truncation
    ):
        self.model = model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.ctx_window = ctx_window
        self.total_calls = 0
        self.deepest_depth_used = 0
    
    def _log(self, message: str, depth: int = 0):
        if self.verbose:
            prefix = "  " * depth
            print(f"{prefix}[ReCAP] {message}")
    
    def _get_depth(self, node: Node) -> int:
        """Calculate depth from root"""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    # ========== Prompt Generation (aligned with ReCAP) ==========
    
    def _build_system_prompt(self) -> str:
        return """You are a math problem solver using recursive task decomposition.

Your approach:
1. Think: Analyze the current task
2. Decompose: Break into subtasks if needed
3. The first subtask will be executed automatically

Response format (JSON):
{
  "think": "Your reasoning about the task",
  "subtasks": ["subtask1", "subtask2", ...]
}

Rules:
- If you can solve directly: put answer in "think" with \\boxed{answer}, set subtasks to []
- If decomposition needed: list subtasks in order
- First subtask will be executed; remaining subtasks are for planning
- When task is complete, return empty subtasks []

Example direct answer: {"think": "Computing: 2+2=4. The answer is \\boxed{4}", "subtasks": []}
"""

    def _generate_init_prompt(self, task: str) -> str:
        return f"""{self._build_system_prompt()}

Your current task: {task}

Please analyze and respond in JSON format.
"""

    def _generate_down_prompt(self, task_name: str) -> str:
        """Prompt when descending to a subtask"""
        return f"""OK.

Your current task: {task_name}

Please generate subtasks for this task, or solve directly if possible.
"""

    def _generate_up_prompt(
        self,
        done_task_name: str,
        previous_stage_task_name: str,
        previous_stage_think: str,
        remaining_subtasks: List[str],
    ) -> str:
        """Prompt when returning from completed subtask (structured re-injection)"""
        if remaining_subtasks:
            remaining_str = '\n'.join(f"- {s}" for s in remaining_subtasks)
        else:
            remaining_str = "No remaining subtasks."
        
        return f"""You have successfully completed the task: {done_task_name}

Now, you return to the parent task.
Your current task: {previous_stage_task_name}

Your previous think: {previous_stage_think}

Your remaining subtasks:
{remaining_str}

Please refine your subtasks based on progress. If task is complete, return empty subtasks [].
"""

    def _generate_judge_done_prompt(
        self,
        done_task_name: str,
        previous_stage_task_name: str,
        previous_stage_think: str,
    ) -> str:
        """Prompt when no remaining subtasks - judge if done"""
        return f"""You have successfully completed the task: {done_task_name}

Now, you return to the parent task.
Your current task: {previous_stage_task_name}

Your previous think: {previous_stage_think}

There are no remaining subtasks. Please determine whether the task is complete.
- If complete: return empty subtasks []
- If not complete: generate necessary subtasks
"""

    def _parse_response(self, response: str) -> dict:
        """Parse JSON response"""
        try:
            response = response.strip()
            # Remove markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback
            return {"think": response, "subtasks": []}

    def _is_leaf_answer(self, subtasks: List[str]) -> bool:
        """Check if subtasks indicate a direct answer (leaf node)"""
        if not subtasks:
            return False
        # Single subtask with boxed answer is a leaf
        if len(subtasks) == 1:
            s = subtasks[0]
            if "\\boxed" in s or "boxed{" in s:
                return True
        return False

    def _has_boxed_answer(self, think: str) -> bool:
        """Check if think contains boxed answer"""
        return "\\boxed" in think or "boxed{" in think

    async def run(self, task: str) -> dict:
        """
        Main loop - aligned with ReCAP's chatbot() function
        Single loop with node pointer movement
        """
        self._log(f"Starting task: {task[:50]}...")
        
        # Initialize
        root = Node(task_name=task)
        node_ptr = root
        during_down = True  # True = descending, False = ascending
        history = []  # Chat history
        iterations = []
        
        # Initial prompt
        prompt = self._generate_init_prompt(task)
        
        for invoke_cnt in range(self.max_iterations):
            if node_ptr is None:
                break
            
            current_depth = self._get_depth(node_ptr)
            self.deepest_depth_used = max(self.deepest_depth_used, current_depth)
            
            # Check depth limit
            if current_depth >= self.max_depth:
                self._log(f"Max depth reached at: {node_ptr.task_name[:30]}...", current_depth)
                # Force completion at max depth
                prompt = f"You are at maximum depth. Please solve directly: {node_ptr.task_name}\nRespond with {{\"think\": \"your answer with \\\\boxed{{}}\", \"subtasks\": []}}"
            
            # Context truncation (aligned with ReCAP: delete middle)
            if len(history) > self.ctx_window:
                # Delete history[2:4] - remove oldest exchanges after system
                del history[2:4]
                self._log("Truncated history (deleted middle)", current_depth)
            
            # Add user prompt
            history.append({"role": "user", "content": prompt})
            
            # Call LLM
            self.total_calls += 1
            self._log(f"Invoke #{invoke_cnt + 1}, depth={current_depth}", current_depth)
            
            response, _ = await gen(
                messages=history,
                model=self.model,
                temperature=0.7,
                response_format="json_object"
            )
            
            if not response:
                self._log("No response from LLM", current_depth)
                break
            
            history.append({"role": "assistant", "content": response})
            
            # Parse response
            parsed = self._parse_response(response)
            think = parsed.get("think") or ""
            subtasks = parsed.get("subtasks") or []
            
            self._log(f"Think: {think[:80]}...", current_depth)
            self._log(f"Subtasks: {subtasks}", current_depth)
            
            # Store info in node
            node_ptr.set_info(parsed)
            
            # Record iteration
            iterations.append({
                "invoke": invoke_cnt + 1,
                "depth": current_depth,
                "task": node_ptr.task_name[:100],
                "think": think[:200],
                "subtasks": subtasks,
                "direction": "down" if during_down else "up"
            })
            
            # ========== State Machine (aligned with ReCAP) ==========
            
            # Case 1: Leaf node - task complete or direct answer
            if not subtasks or self._has_boxed_answer(think):
                self._log(f"Task complete: {node_ptr.task_name[:30]}...", current_depth)
                
                if node_ptr.parent is None:
                    # Root complete
                    self._log("Root task complete!", 0)
                    return {
                        "content": think,
                        "stopped_reason": "complete",
                        "iterations": iterations,
                        "deepest_depth_used": self.deepest_depth_used,
                    }
                
                # Ascend to parent
                during_down = False
                done_task_name = node_ptr.task_name
                node_ptr = node_ptr.parent
                
                if node_ptr is None:
                    break
                
                # Generate up prompt
                parent_info = node_ptr.get_latest_info()
                if parent_info and len(parent_info.get("subtasks", [])) > 1:
                    # Has remaining subtasks
                    prompt = self._generate_up_prompt(
                        done_task_name=done_task_name,
                        previous_stage_task_name=node_ptr.task_name,
                        previous_stage_think=parent_info.get("think", ""),
                        remaining_subtasks=parent_info.get("subtasks", [])[1:],
                    )
                else:
                    # No remaining subtasks - judge done
                    prompt = self._generate_judge_done_prompt(
                        done_task_name=done_task_name,
                        previous_stage_task_name=node_ptr.task_name,
                        previous_stage_think=parent_info.get("think", "") if parent_info else "",
                    )
            
            # Case 2: Has subtasks - descend to first
            else:
                first_subtask = subtasks[0]
                self._log(f"Descending to: {first_subtask[:50]}...", current_depth)
                
                # Create child node
                child = Node(task_name=first_subtask, parent=node_ptr)
                node_ptr.add_child(child)
                node_ptr = child
                during_down = True
                
                # Generate down prompt
                prompt = self._generate_down_prompt(first_subtask)
        
        # Max iterations reached
        self._log("Max iterations reached", 0)
        
        # Try to extract answer from last node
        final_content = None
        if node_ptr:
            info = node_ptr.get_latest_info()
            if info:
                final_content = info.get("think", "")
        
        return {
            "content": final_content,
            "stopped_reason": "max_iterations",
            "iterations": iterations,
            "deepest_depth_used": self.deepest_depth_used,
        }


async def recap_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_iterations: int = 50,
    verbose: bool = True,
    ctx_window: int = 16,
) -> tuple[str, dict]:
    """
    Convenience function for ReCAP agent.
    """
    agent = ReCapAgent(
        model=model,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
        ctx_window=ctx_window,
    )
    
    result = await agent.run(prompt)
    
    if result is None:
        result = {"content": None, "stopped_reason": "error", "iterations": []}
    
    meta = {
        "total_calls": agent.total_calls,
        "max_depth": max_depth,
        "ctx_window": ctx_window,
        "depth_used": result.get("deepest_depth_used", 0),
        "iterations": result.get("iterations", []),
        "stopped_reason": result.get("stopped_reason", "unknown")
    }
    
    return result.get("content"), meta


async def test():
    """Test the ReCAP agent"""
    prompt = "What is 123 * 456 + 789?"
    
    print("=" * 60)
    print("ReCAP Agent Test (Aligned)")
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    content, meta = await recap_gen(
        prompt=prompt,
        model="gpt-4o-mini",
        max_depth=3,
        max_iterations=20,
        verbose=True,
        ctx_window=16,
    )
    
    print("\n" + "=" * 60)
    print(f"Final Answer: {content}")
    print(f"Total calls: {meta['total_calls']}")
    print(f"Depth used: {meta['depth_used']}")
    print(f"Stopped: {meta['stopped_reason']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
