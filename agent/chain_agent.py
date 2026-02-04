"""Chain Agent - Uses invoke to chain multiple LLM calls
Each invoke is a simple LLM query (no recursion), results feed into next iteration
"""

import asyncio
import json
from agent.base_agent import BaseAgent
from api.llm import gen


class ChainAgent(BaseAgent):
    """
    Chain Agent that uses invoke for subtasks.
    
    Each invoke is a simple LLM call - no recursion, just chaining.
    Good for gathering information and step-by-step reasoning.
    """
    
    def get_tools(self) -> dict:
        return {
            "invoke": "Invoke a self-contained subtask to help solve the problem."
        }
    
    def _get_tool_names(self) -> list:
        """Override: allow return on any iteration except force return on last."""
        if self._is_last_iteration():
            return ["return"]
        return ["invoke", "return"]
    
    def _get_all_possible_tools(self) -> list:
        """All tools that could ever be used (for parsing)"""
        return ["invoke", "return"]
    
    def _get_stop_tokens(self) -> list | None:
        """Stop when tool tag closes"""
        return ["</invoke>", "</return>"]
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        return """You are a problem-solving assistant.

You can decompose problems into self-contained subtasks. Each subtask should be independent and well-defined, so it can be solved on its own. By solving these subtasks and combining their results with your own reasoning, you can tackle complex problems step by step.

Subtasks are executed independently - they cannot see your context or the original problem. You must include all necessary information in the subtask description to make it self-contained.

Tools:
- <invoke>your subtask description</invoke> - Create a self-contained subtask. The subtask will be solved and the result returned to you.
- <return>your response</return> - Return your final response.

Think step by step before using tools. Write your reasoning first, then use the appropriate tool.

When returning a final answer, always wrap it in \\boxed{}, e.g., \\boxed{42}.
"""
    
    async def execute_tool(self, name: str, args: dict) -> str | dict:
        if name == "invoke":
            task = args.get("task", "")
            self._log(f"invoke: {task[:50]}...")
            
            # Simple LLM call, no recursion
            response, _ = await gen(prompt=task, model=self.model, temperature=0.7)
            llm_response = response or "No response from LLM"
            
            # Return with llm_call info for detailed logging
            return {
                "result": f"Subtask result: {llm_response}",
                "llm_call": {
                    "task": task,
                    "response": llm_response
                }
            }
        
        return f"Unknown tool: {name}"


async def chain_gen(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_iterations: int = 10,
    verbose: bool = True
) -> tuple[str, dict]:
    """
    Convenience function for chain agent (uses call_llm).
    
    Args:
        prompt: The problem to solve
        model: LLM model to use
        max_iterations: Maximum iterations
        verbose: Whether to print progress
    
    Returns:
        (content, meta) tuple
    """
    agent = ChainAgent(
        model=model,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    result = await agent.run(prompt)
    
    meta = {
        "total_calls": agent.total_calls,
        "tool_calls": result["tool_calls"],
        "iterations": result.get("iterations", []),
        "stopped_reason": result["stopped_reason"]
    }
    
    return result["content"], meta


async def test():
    """Test the chain agent"""
    prompt = "What is 123 * 456 + 789?"
    
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Settings: max_iterations=10")
    print("=" * 60)
    
    content, meta = await chain_gen(
        prompt=prompt,
        model="gpt-4o-mini",
        max_iterations=10,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print(f"Final Answer: {content}")
    print(f"\nMeta: {json.dumps(meta, indent=2, ensure_ascii=False)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
