"""Base Agent - Abstract base class supporting multiple tools
Subclasses define their own tools by overriding get_tools() and execute_tool()
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from api.llm import gen

# No stop tokens - let LLM finish naturally
STOP_TOKENS = None


def parse_response(text: str, tool_names: list) -> dict:
    """Parse LLM response to extract thinking and action
    
    Supports XML tag format: <tool_name>content</tool_name>
    Also handles unclosed tags (when stop tokens are used): <tool_name>content
    
    e.g. <call_agent>solve this problem</call_agent>
         <return>the answer is 42</return>
         <invoke>calculate something   (unclosed, from stop token)
    
    Args:
        text: LLM response text
        tool_names: list of valid tool names
    """
    result = {
        "thinking": None,
        "action": None,
        "args": None,
        "raw": text
    }
    
    if not text:
        return result
    
    # Build regex pattern from tool names for XML tags
    tools_pattern = "|".join(re.escape(name) for name in tool_names)
    
    # Try XML tag format: <tool_name>content</tool_name>
    xml_match = re.search(rf'<({tools_pattern})>(.*?)</\1>', text, re.DOTALL)
    
    if xml_match:
        result["action"] = xml_match.group(1)
        content = xml_match.group(2).strip()
        
        # Map content to appropriate arg key
        if result["action"] == "return":
            result["args"] = {"answer": content}
        elif result["action"] in ("call_agent", "call_llm", "invoke"):
            result["args"] = {"task": content}
        else:
            result["args"] = {"value": content}
        
        # Everything before the tag is thinking
        action_start = xml_match.start()
        if action_start > 0:
            result["thinking"] = text[:action_start].strip()
    else:
        # Try unclosed tag format: <tool_name>content (from stop tokens)
        unclosed_match = re.search(rf'<({tools_pattern})>(.*)$', text, re.DOTALL)
        
        if unclosed_match:
            result["action"] = unclosed_match.group(1)
            content = unclosed_match.group(2).strip()
            
            # Map content to appropriate arg key
            if result["action"] == "return":
                result["args"] = {"answer": content}
            elif result["action"] in ("call_agent", "call_llm", "invoke"):
                result["args"] = {"task": content}
            else:
                result["args"] = {"value": content}
            
            # Everything before the tag is thinking
            action_start = unclosed_match.start()
            if action_start > 0:
                result["thinking"] = text[:action_start].strip()
        else:
            result["thinking"] = text.strip()
    
    return result


def parse_args(args_str: str, tool_name: str = None) -> dict:
    """Parse function arguments string
    
    Supports both:
    - Named args: call_agent(task="...") 
    - Positional args: call_agent("...")
    """
    args = {}
    
    # First try named arguments: key=value
    pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))'
    
    for match in re.finditer(pattern, args_str):
        key = match.group(1)
        value = match.group(2) or match.group(3) or match.group(4) or match.group(5)
        
        if value is not None:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif match.group(4):
                value = float(value) if '.' in value else int(value)
            else:
                value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
            args[key] = value
    
    # If no named args found, try positional string argument
    if not args:
        # Match quoted string as positional argument
        pos_match = re.match(r'^\s*["\'](.+?)["\']\s*$', args_str, re.DOTALL)
        if pos_match:
            value = pos_match.group(1)
            # Map positional arg to appropriate key based on tool
            if tool_name in ("call_agent", "call_llm"):
                args["task"] = value
            elif tool_name == "finish":
                args["answer"] = value
            else:
                args["value"] = value
    
    return args


class BaseAgent(ABC):
    """
    Abstract Base Agent supporting multiple tools.
    
    Subclasses must implement:
    - get_tools(): return dict of {tool_name: tool_description}
    - execute_tool(name, args): execute a tool and return result
    
    Built-in tools:
    - return: always available, returns final response
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.total_calls = 0
        self._current_iteration = 0  # Track current iteration for last-iteration logic
    
    def _log(self, message: str, indent: int = 0):
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}[{self.__class__.__name__}] {message}")
    
    @abstractmethod
    def get_tools(self) -> dict:
        """Return dict of {tool_name: tool_description}
        
        Example:
            return {
                "call_llm": "Ask LLM for information",
                "search": "Search the web for information"
            }
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, name: str, args: dict) -> str | dict:
        """Execute a tool and return result
        
        Args:
            name: tool name
            args: tool arguments dict
        
        Returns:
            result string, OR dict with {"result": str, "sub_agent_info": {...}} for nested logging
        """
        pass
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from tools"""
        tools = self.get_tools()
        
        tools_section = "\n".join([
            f"- <{name}>...</{name}> - {desc}"
            for name, desc in tools.items()
        ])
        
        return f"""You are a problem-solving assistant. Use tools to help you think and solve problems.

Available tools:
{tools_section}
- <return>your response</return> - Return your response when done.

Think step by step before calling tools. Write your reasoning first, then call the appropriate tool.

When returning a final answer, always wrap it in \\boxed{{}}, e.g., \\boxed{{42}}.
"""
    
    def _is_last_iteration(self) -> bool:
        """Check if this is the last iteration"""
        return self._current_iteration >= self.max_iterations - 1
    
    def _get_tool_names(self) -> list:
        """Get list of all valid tool names including return.
        On last iteration, only return is allowed."""
        if self._is_last_iteration():
            return ["return"]  # Force return on last iteration
        return list(self.get_tools().keys()) + ["return"]
    
    async def run(self, prompt: str) -> dict:
        """Run the Agent"""
        self._log(f"Starting")
        
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return await self._agent_loop(messages)
    
    def _get_all_possible_tools(self) -> list:
        """Get all tools that could ever be used (for parsing)
        Override in subclass if some tools are conditionally available"""
        return list(self.get_tools().keys()) + ["return"]
    
    def _get_stop_tokens(self) -> list | None:
        """Get stop tokens for LLM generation.
        Override in subclass to customize stop behavior."""
        return None
    
    async def _agent_loop(self, messages: list) -> dict:
        """Agent main loop"""
        tool_call_history = []
        iteration_history = []
        # Use all possible tools for parsing (so we can detect and reject disallowed tools)
        all_tools = self._get_all_possible_tools()
        
        iteration = 0
        last_iteration_retries = 0
        max_last_iteration_retries = 3  # Max retries on last iteration before giving up
        
        while iteration < self.max_iterations:
            self._current_iteration = iteration
            is_last = self._is_last_iteration()
            self._log(f"Iteration {iteration + 1}/{self.max_iterations}" + (f" [LAST - must return, retry {last_iteration_retries}/{max_last_iteration_retries}]" if is_last else ""))
            self.total_calls += 1
            
            # Call LLM
            response, messages = await gen(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=2048,
                stop=self._get_stop_tokens()
            )
            
            if not response:
                self._log("No response from LLM")
                return {
                    "content": None,
                    "tool_calls": tool_call_history,
                    "iterations": iteration_history,
                    "stopped_reason": "error"
                }
            
            # Parse response using all possible tools
            parsed = parse_response(response, all_tools)
            
            if parsed["thinking"]:
                thinking_preview = parsed["thinking"][:100].replace('\n', ' ')
                self._log(f"Thinking: {thinking_preview}...")
            
            if not parsed["action"]:
                self._log("No action found, reminding LLM to use tools")
                self.total_calls -= 1
                # Get currently available tools for reminder
                available_tools = self._get_tool_names()
                tools_list = ", ".join(f'<{name}>...</{name}>' for name in available_tools)
                reminder = f"You must call a tool. Available: {tools_list}."
                messages.append({"role": "user", "content": reminder})
                # Don't increment iteration on last iteration - keep prompting until return
                if is_last:
                    last_iteration_retries += 1
                    if last_iteration_retries >= max_last_iteration_retries:
                        self._log(f"Max retries on last iteration reached, giving up")
                        break
                else:
                    iteration += 1
                continue
            
            # Check if the action is currently allowed
            allowed_tools = self._get_tool_names()
            if parsed["action"] not in allowed_tools:
                self._log(f"Tool '{parsed['action']}' not available, rejecting")
                self.total_calls -= 1
                tools_list = ", ".join(f'<{name}>...</{name}>' for name in allowed_tools)
                rejection = f"Tool '{parsed['action']}' is not available. You MUST use <return>your final answer</return> now."
                messages.append({"role": "user", "content": rejection})
                # Don't increment iteration on last iteration - keep prompting until return
                if is_last:
                    last_iteration_retries += 1
                    if last_iteration_retries >= max_last_iteration_retries:
                        self._log(f"Max retries on last iteration reached, giving up")
                        break
                else:
                    iteration += 1
                continue
            
            action_name = parsed["action"]
            action_args = parsed["args"] or {}
            
            self._log(f"Action: {action_name}({action_args})")
            
            # Execute action
            if action_name == "return":
                final_answer = str(action_args.get("answer", ""))
                self._log(f"Return: {final_answer[:50]}...")
                
                iteration_history.append({
                    "iteration": iteration + 1,
                    "agent_response": response,
                    "thinking": parsed["thinking"],
                    "action": action_name,
                    "args": action_args,
                    "result": final_answer
                })
                
                tool_call_history.append({
                    "name": action_name,
                    "args": action_args,
                    "result": final_answer
                })
                
                return {
                    "content": final_answer,
                    "tool_calls": tool_call_history,
                    "iterations": iteration_history,
                    "stopped_reason": "return"
                }
            
            elif action_name in self.get_tools():
                tool_result = await self.execute_tool(action_name, action_args)
                
                # Handle both simple string result and dict with extra info
                extra_info = {}
                if isinstance(tool_result, dict):
                    result = tool_result.get("result", "")
                    # Capture any extra info (sub_agent_info, llm_call, etc.)
                    for key in tool_result:
                        if key != "result":
                            extra_info[key] = tool_result[key]
                else:
                    result = tool_result
            
            else:
                result = f"Unknown tool: {action_name}. Available: {', '.join(all_tools)}"
                extra_info = {}
            
            self._log(f"Result: {str(result)[:80]}...")
            
            # Record iteration with optional extra info (sub_agent, llm_call, etc.)
            iteration_record = {
                "iteration": iteration + 1,
                "agent_response": response,
                "thinking": parsed["thinking"],
                "action": action_name,
                "args": action_args,
                "result": result
            }
            # Add any extra info to the record
            iteration_record.update(extra_info)
            iteration_history.append(iteration_record)
            
            tool_call_history.append({
                "name": action_name,
                "args": action_args,
                "result": ("..." + str(result)[-200:]) if result and len(str(result)) > 200 else (str(result) if result else None)
            })
            
            # Add observation
            observation_content = f"<observation>\n{result}\n</observation>"
            
            # Warn agent on last iteration
            remaining = self.max_iterations - iteration - 1
            if remaining == 1:
                observation_content += "\n\nThis is your LAST iteration. You MUST use <return>your final answer</return> now. Put your numeric answer in \\boxed{}."
            
            messages.append({"role": "user", "content": observation_content})
            iteration += 1
        
        # Reached max_iterations without proper return - try to extract answer from history
        fallback_answer = self._extract_fallback_answer(iteration_history, messages)
        
        return {
            "content": fallback_answer,
            "tool_calls": tool_call_history,
            "iterations": iteration_history,
            "stopped_reason": "max_iterations"
        }
    
    def _extract_fallback_answer(self, iteration_history: list, messages: list) -> str | None:
        """Try to extract a fallback answer when max_iterations is reached.
        
        Strategy:
        1. Look for \\boxed{} in any iteration's response or result
        2. Look for the last subtask result that looks like an answer
        3. Look for any numeric answer in recent responses
        """
        import re
        
        # Strategy 1: Find \\boxed{} in iteration history (reverse order - prefer recent)
        for iteration in reversed(iteration_history):
            # Check agent response
            response = iteration.get("agent_response", "")
            boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
            if boxed:
                return boxed[-1].strip()
            
            # Check result (from subtasks)
            result = str(iteration.get("result", ""))
            boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', result)
            if boxed:
                return boxed[-1].strip()
        
        # Strategy 2: Look for subtask results that contain answers
        for iteration in reversed(iteration_history):
            result = str(iteration.get("result", ""))
            if "Subtask result:" in result:
                # Extract the actual result content
                content = result.replace("Subtask result:", "").strip()
                # Look for boxed in subtask result
                boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', content)
                if boxed:
                    return boxed[-1].strip()
        
        # Strategy 3: Extract from last few messages
        for msg in reversed(messages[-6:]):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', content)
                if boxed:
                    return boxed[-1].strip()
        
        return None
