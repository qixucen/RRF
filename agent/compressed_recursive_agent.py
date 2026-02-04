"""Compressed Recursive Agent - Recursive reasoning with world state maintenance

Introduces additional LLM calls at invoke and return to manage context information,
binding information lifecycle to recursive structure to avoid infinite accumulation.
"""

import asyncio
import json
from agent.base_agent import BaseAgent
from api.llm import gen


class CompressedRecursiveAgent(BaseAgent):
    """
    Recursive Agent with world state maintenance.
    
    Key improvements:
    1. Before invoke: Use LLM to refine context, decide what info to pass to child task
    2. After return: Use LLM to compress child result, keep only key information
    
    This way, information lifecycle is bound to recursive structure - detailed context
    from child tasks is compressed when returning to parent, not accumulated infinitely.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_depth: int = 3,
        max_iterations: int = 10,
        verbose: bool = True,
        force_invoke: bool = False,
        _current_depth: int = None,
        _parent_context: str = None  # Refined context from parent task
    ):
        super().__init__(model=model, max_iterations=max_iterations, verbose=verbose)
        self.max_depth = max_depth
        self.force_invoke = force_invoke
        self._current_depth = _current_depth if _current_depth is not None else max_depth
        self._parent_context = _parent_context
        self.deepest_depth_used = 0
        self._invoke_count = 0
        self._world_state_calls = 0  # Track world state maintenance calls
    
    def _log(self, message: str, indent: int = 0):
        if self.verbose:
            prefix = "  " * (self.max_depth - self._current_depth)
            print(f"{prefix}[D{self._current_depth}] {message}")
    
    def _get_tool_names(self) -> list:
        """Override: Decide available tools based on conditions"""
        if self._is_last_iteration():
            return ["return"]
        
        tools = list(self.get_tools().keys())
        if not self.force_invoke or self._invoke_count >= self._current_depth:
            tools.append("return")
        return tools
    
    def _get_all_possible_tools(self) -> list:
        return ["invoke", "return"]
    
    def _get_stop_tokens(self) -> list | None:
        return ["</invoke>", "</return>"]
    
    def get_tools(self) -> dict:
        return {
            "invoke": "Invoke a self-contained subtask to help solve the problem."
        }
    
    async def _prepare_context_for_child(self, task: str, current_messages: list) -> str:
        """
        World State Maintenance - Phase 1: Prepare context to pass to child task
        
        Use LLM to extract helpful information from current conversation history,
        rather than passing all information down.
        """
        self._log("WorldState: Preparing context for child...")
        self._world_state_calls += 1
        
        # Build current conversation summary
        conversation_summary = []
        for msg in current_messages[-6:]:  # Only look at recent messages
            role = msg.get("role", "")
            content = msg.get("content", "")[:500]  # Truncate
            if role in ("user", "assistant"):
                conversation_summary.append(f"[{role}]: {content}")
        
        prompt = f"""You are a context management assistant. The main task is decomposing subtasks, and you need to decide what information to pass to the subtask.

Current conversation history summary:
{chr(10).join(conversation_summary)}

Subtask description:
{task}

Please extract the most helpful context information for this subtask (if any). Requirements:
1. Only extract information directly related to the subtask
2. Be concise, no more than 100 words
3. If no context needs to be passed, output "None"

Output format: Directly output the refined context without extra formatting."""
        
        response, _ = await gen(
            prompt=prompt,
            model=self.model,
            temperature=0.3,
            max_tokens=200
        )
        
        context = (response or "").strip()
        if context and context.lower() != "none":
            self._log(f"WorldState: Context prepared: {context[:50]}...")
            return context
        return None
    
    async def _compress_child_result(self, task: str, result: str) -> str:
        """
        World State Maintenance - Phase 2: Compress child task result
        
        Child tasks may return verbose reasoning processes, we only need to keep key results,
        letting information naturally decay with recursive levels.
        """
        # If result is short, no need to compress
        if len(result) < 300:
            return result
        
        self._log("WorldState: Compressing child result...")
        self._world_state_calls += 1
        
        prompt = f"""You are a result compression assistant. A subtask has completed, and you need to compress its result.

Subtask:
{task[:200]}

Complete result from subtask:
{result}

Please extract the most critical information from this result. Requirements:
1. Keep the core answer/conclusion
2. Keep key intermediate results (if there are numerical calculations)
3. Remove redundant reasoning process
4. Compressed result should not exceed 150 words

Output format: Directly output the compressed result without extra formatting."""
        
        response, _ = await gen(
            prompt=prompt,
            model=self.model,
            temperature=0.3,
            max_tokens=300
        )
        
        compressed = (response or result).strip()
        self._log(f"WorldState: Compressed {len(result)} -> {len(compressed)} chars")
        return compressed
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        if self.force_invoke:
            remaining = self._current_depth - self._invoke_count
            if remaining > 0:
                return_status = f"(locked, need {remaining} more invoke to unlock)"
            else:
                return_status = "(unlocked)"
        else:
            return_status = ""
        
        # If there's context from parent task, add to prompt
        context_section = ""
        if self._parent_context:
            context_section = f"""
Important context from parent task:
{self._parent_context}

Use this context to help you solve the problem.
"""
        
        return f"""You are a problem-solving assistant.
{context_section}
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
            self._invoke_count += 1
            
            if self._current_depth > 0:
                # === World State Maintenance Phase 1: Prepare child task context ===
                child_context = await self._prepare_context_for_child(
                    task, 
                    getattr(self, '_current_messages', [])
                )
                
                # Create child agent, pass refined context
                sub_agent = CompressedRecursiveAgent(
                    model=self.model,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations,
                    verbose=self.verbose,
                    force_invoke=self.force_invoke,
                    _current_depth=self._current_depth - 1,
                    _parent_context=child_context  # Pass refined context
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
                        "raw_content": raw_content,  # Original result
                        "compressed_content": compressed_content,  # Compressed result
                        "parent_context": child_context,  # Context passed to child
                        "iterations": result.get("iterations", []),
                        "deepest_depth_used": sub_agent.deepest_depth_used
                    }
                }
            else:
                # Deepest level: Direct LLM call
                self._log(f"invoke (direct): {task[:50]}...")
                
                # If there's parent context, add to prompt
                if self._parent_context:
                    enhanced_task = f"Context: {self._parent_context}\n\nTask: {task}"
                else:
                    enhanced_task = task
                
                response, _ = await gen(prompt=enhanced_task, model=self.model, temperature=0.7)
                llm_response = response or "No response"
                
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
        
        # Save messages reference for execute_tool to use
        self._current_messages = messages
        
        result = await self._agent_loop(messages)
        result["depth"] = self._current_depth
        result["deepest_depth_used"] = self.deepest_depth_used
        result["world_state_calls"] = self._world_state_calls
        return result


async def compressed_recursive_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_iterations: int = 10,
    verbose: bool = True,
    force_invoke: bool = False
) -> tuple[str, dict]:
    """
    Recursive reasoning with world state maintenance.
    
    Args:
        prompt: The problem to solve
        model: LLM model to use
        max_depth: Maximum depth for subtask decomposition
        max_iterations: Maximum iterations per agent
        verbose: Whether to print progress
        force_invoke: If True, must invoke enough subtasks to unlock return.
    
    Returns:
        (content, meta) tuple
    """
    agent = CompressedRecursiveAgent(
        model=model,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
        force_invoke=force_invoke
    )
    
    result = await agent.run(prompt)
    
    meta = {
        "total_calls": agent.total_calls,
        "world_state_calls": agent._world_state_calls,
        "max_depth": max_depth,
        "force_invoke": force_invoke,
        "depth_used": result.get("deepest_depth_used", 0),
        "tool_calls": result["tool_calls"],
        "iterations": result.get("iterations", []),
        "stopped_reason": result["stopped_reason"]
    }
    
    return result["content"], meta