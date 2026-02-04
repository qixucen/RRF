"""Invoke Agent - Solves problems by decomposing into self-contained subtasks
Each subtask can also be further decomposed, with depth control
"""

import asyncio
import json
from agent.base_agent import BaseAgent
from api.llm import gen


class RecursiveAgent(BaseAgent):
    """
    Invoke Agent that solves problems by decomposing into self-contained subtasks.
    
    Uses <invoke> to create subtasks, with configurable depth levels.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_depth: int = 3,
        max_iterations: int = 10,
        verbose: bool = True,
        force_invoke: bool = False,  # If True, must invoke enough times to unlock return
        _current_depth: int = None  # Internal: current depth level
    ):
        super().__init__(model=model, max_iterations=max_iterations, verbose=verbose)
        self.max_depth = max_depth
        self.force_invoke = force_invoke
        self._current_depth = _current_depth if _current_depth is not None else max_depth
        self.deepest_depth_used = 0  # Track how many levels actually used
        self._invoke_count = 0  # Count invoke calls
    
    def _log(self, message: str, indent: int = 0):
        if self.verbose:
            prefix = "  " * (self.max_depth - self._current_depth)
            print(f"{prefix}[D{self._current_depth}] {message}")
    
    def _get_tool_names(self) -> list:
        """Override: need to invoke enough times to unlock return (if force_invoke=True).
        On last iteration, only return is allowed."""
        # Force return on last iteration
        if self._is_last_iteration():
            return ["return"]
        
        tools = list(self.get_tools().keys())
        # If force_invoke=False, always allow return
        # If force_invoke=True, can return only if invoked enough times (>= current_depth)
        if not self.force_invoke or self._invoke_count >= self._current_depth:
            tools.append("return")
        return tools
    
    def _get_all_possible_tools(self) -> list:
        """All tools that could ever be used (for parsing)"""
        return ["invoke", "return"]
    
    def _get_stop_tokens(self) -> list | None:
        """Stop when tool tag closes"""
        return ["</invoke>", "</return>"]
    
    def get_tools(self) -> dict:
        return {
            "invoke": "Invoke a self-contained subtask to help solve the problem."
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        if self.force_invoke:
            remaining = self._current_depth - self._invoke_count
            if remaining > 0:
                return_status = f"(locked, need {remaining} more invoke to unlock)"
            else:
                return_status = "(unlocked)"
        else:
            return_status = ""  # No status needed when return is always available
        
        return f"""You are a problem-solving assistant.

You can decompose problems into self-contained subtasks. Each subtask should be independent and well-defined, so it can be solved on its own. By solving these subtasks and combining their results with your own reasoning, you can tackle complex problems step by step.

Subtasks are executed independently - they cannot see your context or the original problem. You must include all necessary information in the subtask description to make it self-contained.

Tools:
- <invoke>your subtask description</invoke> - Create a self-contained subtask. The subtask will be solved and the result returned to you.
- <return>your response</return> - Return your final response. {return_status}

Think step by step before using tools. Write your reasoning first, then use the appropriate tool.

When returning a final answer, always wrap it in \\boxed{{}}, e.g., \\boxed{{42}}.
"""
    
    async def execute_tool(self, name: str, args: dict) -> str | dict:
        task = args.get("task", "")
        
        if name == "invoke":
            self._log(f"invoke: {task[:50]}...")
            self._invoke_count += 1  # Increment counter
            
            if self._current_depth > 0:
                # Create sub-agent with depth - 1
                sub_agent = RecursiveAgent(
                    model=self.model,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations,
                    verbose=self.verbose,
                    force_invoke=self.force_invoke,
                    _current_depth=self._current_depth - 1
                )
                
                result = await sub_agent.run(task)
                
                # Add sub-agent's calls to our total
                self.total_calls += sub_agent.total_calls
                
                # Track deepest level: 1 (this call) + sub-agent's depth
                sub_depth = 1 + sub_agent.deepest_depth_used
                self.deepest_depth_used = max(self.deepest_depth_used, sub_depth)
                
                # Build result with sub_agent_info for detailed logging
                if self.force_invoke:
                    remaining = self._current_depth - self._invoke_count
                    if remaining > 0:
                        unlock_hint = f"[Invoke {remaining} more subtask(s) to unlock return]"
                    else:
                        unlock_hint = "[return is now unlocked. You can use <return>your response</return> when ready, or continue invoking subtasks if needed.]"
                else:
                    unlock_hint = ""  # No hint needed when return is always available
                
                if result["content"]:
                    result_str = f"Subtask result: {result['content']}\n\n{unlock_hint}"
                else:
                    result_str = f"Subtask failed: {result['stopped_reason']}\n\n{unlock_hint}"
                
                return {
                    "result": result_str,
                    "subtask_info": {
                        "depth": self._current_depth - 1,
                        "total_calls": sub_agent.total_calls,
                        "stopped_reason": result["stopped_reason"],
                        "content": result["content"],
                        "iterations": result.get("iterations", []),
                        "deepest_depth_used": sub_agent.deepest_depth_used
                    }
                }
            else:
                # Deepest level: directly call LLM
                self._log(f"invoke (direct): {task[:50]}...")
                response, _ = await gen(prompt=task, model=self.model, temperature=0.7)
                llm_response = response or "No response"
                
                # Return with llm_call info for detailed logging
                return {
                    "result": f"Subtask result: {llm_response}",
                    "llm_call": {
                        "task": task,
                        "response": llm_response
                    }
                }
        
        return f"Unknown tool: {name}"
    
    async def run(self, prompt: str) -> dict:
        """Run the Agent"""
        self._log(f"Starting (depth={self._current_depth}/{self.max_depth})")
        
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        result = await self._agent_loop(messages)
        result["depth"] = self._current_depth
        result["deepest_depth_used"] = self.deepest_depth_used
        return result


async def recursive_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_iterations: int = 10,
    verbose: bool = True,
    force_invoke: bool = False
) -> tuple[str, dict]:
    """
    Convenience function for invoke-based LLM agent.
    
    Args:
        prompt: The problem to solve
        model: LLM model to use
        max_depth: Maximum depth for subtask decomposition
        max_iterations: Maximum iterations per agent
        verbose: Whether to print progress
        force_invoke: If True, must invoke enough subtasks to unlock return.
                     If False (default), agent can choose invoke or return freely.
    
    Returns:
        (content, meta) tuple
    """
    agent = RecursiveAgent(
        model=model,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
        force_invoke=force_invoke
    )
    
    result = await agent.run(prompt)
    
    meta = {
        "total_calls": agent.total_calls,
        "max_depth": max_depth,
        "force_invoke": force_invoke,
        "depth_used": result.get("deepest_depth_used", 0),  # Actual depth levels used
        "tool_calls": result["tool_calls"],
        "iterations": result.get("iterations", []),
        "stopped_reason": result["stopped_reason"]
    }
    
    return result["content"], meta