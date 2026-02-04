import asyncio
import fcntl
import json
import os
import random
import time
from pathlib import Path

import openai
import yaml


# ============================================================================
# 跨进程 LLM 调用监控器
# ============================================================================

class CrossProcessLLMMonitor:
    """
    跨进程 LLM 调用监控器
    
    使用文件锁 + JSON 状态文件实现跨进程的并发监控和控制。
    无论运行多少个 Python 脚本，都能统一追踪系统中正在进行的 LLM 调用数量。
    
    工作原理：
    1. 使用一个 JSON 文件记录所有活跃的 LLM 调用
    2. 使用文件锁（fcntl）保证并发安全
    3. 每个调用开始时注册，结束时注销
    4. 定期清理超时的僵尸记录
    """
    
    # 默认状态文件路径
    DEFAULT_STATE_FILE = "/tmp/llm_monitor_state.json"
    DEFAULT_LOCK_FILE = "/tmp/llm_monitor.lock"
    
    # 调用超时时间（秒），超过这个时间的记录会被清理
    CALL_TIMEOUT = 600  # 10 分钟
    
    # 清理间隔（秒）
    CLEANUP_INTERVAL = 30
    
    _instance = None
    _last_cleanup_time = 0
    
    def __new__(cls, state_file=None, lock_file=None):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, state_file=None, lock_file=None):
        if self._initialized:
            return
        
        self._state_file = Path(state_file or self.DEFAULT_STATE_FILE)
        self._lock_file = Path(lock_file or self.DEFAULT_LOCK_FILE)
        self._pid = os.getpid()
        self._call_counter = 0  # 进程内调用计数器
        self._initialized = True
        
        # 确保状态文件存在
        self._ensure_state_file()
    
    def _ensure_state_file(self):
        """确保状态文件存在"""
        if not self._state_file.exists():
            self._write_state({"calls": {}, "stats": {"total_calls": 0}})
    
    def _read_state(self) -> dict:
        """读取状态（需要在锁内调用）"""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {"calls": {}, "stats": {"total_calls": 0}}
    
    def _write_state(self, state: dict):
        """写入状态（需要在锁内调用）"""
        with open(self._state_file, 'w') as f:
            json.dump(state, f)
    
    def _with_lock(self, func):
        """使用文件锁执行操作"""
        # 确保锁文件存在
        self._lock_file.touch(exist_ok=True)
        
        with open(self._lock_file, 'r+') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return func()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    
    def _cleanup_stale_calls(self, state: dict) -> dict:
        """清理超时的僵尸调用记录"""
        now = time.time()
        calls = state.get("calls", {})
        active_calls = {}
        
        for call_id, call_info in calls.items():
            start_time = call_info.get("start_time", 0)
            if now - start_time < self.CALL_TIMEOUT:
                active_calls[call_id] = call_info
        
        state["calls"] = active_calls
        return state
    
    def _generate_call_id(self) -> str:
        """生成唯一的调用 ID"""
        self._call_counter += 1
        return f"{self._pid}_{self._call_counter}_{time.time()}"
    
    def register_call(self, model: str = "unknown") -> str:
        """
        注册一个新的 LLM 调用
        
        Returns:
            call_id: 用于后续注销的唯一标识
        """
        call_id = self._generate_call_id()
        
        def _do_register():
            state = self._read_state()
            
            # 定期清理
            now = time.time()
            if now - CrossProcessLLMMonitor._last_cleanup_time > self.CLEANUP_INTERVAL:
                state = self._cleanup_stale_calls(state)
                CrossProcessLLMMonitor._last_cleanup_time = now
            
            # 注册新调用
            state["calls"][call_id] = {
                "pid": self._pid,
                "model": model,
                "start_time": now,
            }
            state["stats"]["total_calls"] = state["stats"].get("total_calls", 0) + 1
            
            self._write_state(state)
            return len(state["calls"])
        
        current_count = self._with_lock(_do_register)
        return call_id
    
    def unregister_call(self, call_id: str):
        """注销一个 LLM 调用"""
        def _do_unregister():
            state = self._read_state()
            if call_id in state.get("calls", {}):
                del state["calls"][call_id]
                self._write_state(state)
        
        self._with_lock(_do_unregister)
    
    def get_active_count(self) -> int:
        """获取当前活跃的 LLM 调用数量"""
        def _do_count():
            state = self._read_state()
            state = self._cleanup_stale_calls(state)
            self._write_state(state)
            return len(state.get("calls", {}))
        
        return self._with_lock(_do_count)
    
    def get_status(self) -> dict:
        """
        获取详细状态信息
        
        Returns:
            {
                "active_calls": 当前活跃调用数,
                "by_process": {pid: count, ...},
                "by_model": {model: count, ...},
                "total_calls": 历史总调用数,
                "calls_detail": [{pid, model, duration}, ...]
            }
        """
        def _do_get_status():
            state = self._read_state()
            state = self._cleanup_stale_calls(state)
            self._write_state(state)
            
            calls = state.get("calls", {})
            now = time.time()
            
            by_process = {}
            by_model = {}
            calls_detail = []
            
            for call_id, info in calls.items():
                pid = info.get("pid", "unknown")
                model = info.get("model", "unknown")
                duration = now - info.get("start_time", now)
                
                by_process[pid] = by_process.get(pid, 0) + 1
                by_model[model] = by_model.get(model, 0) + 1
                calls_detail.append({
                    "pid": pid,
                    "model": model,
                    "duration": round(duration, 1),
                })
            
            return {
                "active_calls": len(calls),
                "by_process": by_process,
                "by_model": by_model,
                "total_calls": state.get("stats", {}).get("total_calls", 0),
                "calls_detail": sorted(calls_detail, key=lambda x: -x["duration"]),
            }
        
        return self._with_lock(_do_get_status)
    
    def wait_if_too_many(self, max_concurrent: int, check_interval: float = 0.5) -> int:
        """
        如果当前并发数过高，等待直到降到阈值以下
        
        Args:
            max_concurrent: 最大允许并发数
            check_interval: 检查间隔（秒）
        
        Returns:
            当前活跃调用数
        """
        while True:
            count = self.get_active_count()
            if count < max_concurrent:
                return count
            time.sleep(check_interval)
    
    async def wait_if_too_many_async(self, max_concurrent: int, check_interval: float = 0.5) -> int:
        """异步版本的等待"""
        while True:
            count = self.get_active_count()
            if count < max_concurrent:
                return count
            await asyncio.sleep(check_interval)


# 全局监控器实例
_global_monitor = None

def get_llm_monitor() -> CrossProcessLLMMonitor:
    """获取全局 LLM 监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CrossProcessLLMMonitor()
    return _global_monitor


def print_llm_status():
    """打印当前 LLM 调用状态（方便调试）"""
    monitor = get_llm_monitor()
    status = monitor.get_status()
    
    print("\n" + "=" * 50)
    print(f"🔥 LLM 调用监控状态")
    print("=" * 50)
    print(f"当前活跃调用: {status['active_calls']}")
    print(f"历史总调用数: {status['total_calls']}")
    
    if status['by_process']:
        print(f"\n按进程分布:")
        for pid, count in sorted(status['by_process'].items(), key=lambda x: -x[1]):
            print(f"  PID {pid}: {count} 个调用")
    
    if status['by_model']:
        print(f"\n按模型分布:")
        for model, count in sorted(status['by_model'].items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} 个调用")
    
    if status['calls_detail']:
        print(f"\n活跃调用详情 (按时长排序):")
        for i, call in enumerate(status['calls_detail'][:10]):  # 最多显示10个
            print(f"  [{i+1}] PID={call['pid']}, model={call['model']}, duration={call['duration']}s")
        if len(status['calls_detail']) > 10:
            print(f"  ... 还有 {len(status['calls_detail']) - 10} 个调用")
    
    print("=" * 50 + "\n")


# ============================================================================
# 自适应速率限制器（支持跨进程感知）
# ============================================================================

class AdaptiveRateLimiter:
    """
    自适应速率控制器 - 自动探测最大可用速率和最大并发数
    
    采用类似 TCP 拥塞控制的 AIMD 策略：
    - 成功时线性增加速率和并发上限（Additive Increase）
    - 遇到 rate limit 时乘法降低（Multiplicative Decrease）
    
    同时维护当前正在执行的任务数（in-flight count）。
    如果并发数未达上限，直接放行；达到上限后才启用速率限制。
    """
    
    def __init__(
        self,
        initial_rps: float = 50,
        min_rps: float = 5,
        max_concurrent: int = 500,        # 初始最大并发数
        min_concurrent: int = 50,         # 最小并发数
        burst_allowance: float = 2.0,
        increase_step: float = 5.0,       # 每次增加的 RPS
        concurrent_increase_step: int = 10,  # 每次增加的并发数
        increase_interval: int = 20,      # 每多少次成功后增加
        decrease_factor: float = 0.5,     # 遇到 rate limit 时乘以的系数
    ):
        self._rps = initial_rps
        self._min_rps = min_rps
        self._max_concurrent = max_concurrent
        self._min_concurrent = min_concurrent
        self._burst_allowance = burst_allowance
        self._increase_step = increase_step
        self._concurrent_increase_step = concurrent_increase_step
        self._increase_interval = increase_interval
        self._decrease_factor = decrease_factor
        
        # Token bucket 状态
        self._tokens = initial_rps * burst_allowance
        self._last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()
        
        # 并发任务计数
        self._in_flight = 0
        self._peak_in_flight = 0
        
        # 统计
        self._rate_limit_count = 0
        self._success_count = 0
        self._success_since_last_limit = 0
        
        # 压力检测：只有最近有高负载时才增速
        self._last_pressure_time = 0  # 最近一次高负载的时间
        self._pressure_threshold = 0.5  # 超过 50% 占用率算有压力
        self._pressure_window = 10.0  # 压力有效窗口（秒）
    
    async def acquire(self):
        """获取令牌（控制发送速率）"""
        async with self._lock:
            # 检测压力：如果当前负载较高，记录压力时间
            occupancy = self._in_flight / self._max_concurrent if self._max_concurrent > 0 else 0
            if occupancy >= self._pressure_threshold:
                self._last_pressure_time = time.monotonic()
            
            # 如果并发数未达上限，直接放行
            if self._in_flight < self._max_concurrent:
                self._in_flight += 1
                if self._in_flight > self._peak_in_flight:
                    self._peak_in_flight = self._in_flight
                return
            
            # 达到并发上限，使用 token bucket 控制速率
            now = time.monotonic()
            elapsed = now - self._last_refill_time
            
            # 补充令牌
            max_tokens = self._rps * self._burst_allowance
            self._tokens = min(max_tokens, self._tokens + elapsed * self._rps)
            self._last_refill_time = now
            
            # 如果没有令牌，等待
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rps
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1
            
            # 增加并发计数
            self._in_flight += 1
            if self._in_flight > self._peak_in_flight:
                self._peak_in_flight = self._in_flight
    
    def release(self):
        """释放，减少并发计数"""
        self._in_flight = max(0, self._in_flight - 1)
    
    def report_rate_limit(self):
        """
        报告 rate limit，智能降速
        
        根据当前 in_flight 占 max_concurrent 的比例判断瓶颈：
        - 高占比（>70%）：主要降低 max_concurrent，轻微降低 RPS
        - 低占比（<30%）：主要降低 RPS，轻微降低 max_concurrent
        - 中等占比：两者均衡降低
        """
        self._rate_limit_count += 1
        old_rps = self._rps
        old_concurrent = self._max_concurrent
        
        # 计算 in_flight 占比
        occupancy = self._in_flight / self._max_concurrent if self._max_concurrent > 0 else 0
        
        if occupancy > 0.7:
            # 并发数可能是瓶颈，主要降低并发
            concurrent_factor = 0.7  # 降 30%
            rps_factor = 0.9         # 降 10%
            reason = "concurrent bottleneck"
        elif occupancy < 0.3:
            # RPS 可能是瓶颈，主要降低 RPS
            concurrent_factor = 0.9  # 降 10%
            rps_factor = 0.7         # 降 30%
            reason = "RPS bottleneck"
        else:
            # 不确定，均衡降低
            concurrent_factor = 0.8  # 降 20%
            rps_factor = 0.8         # 降 20%
            reason = "balanced"
        
        self._rps = max(self._min_rps, self._rps * rps_factor)
        self._max_concurrent = max(self._min_concurrent, int(self._max_concurrent * concurrent_factor))
        self._success_since_last_limit = 0  # 重置成功计数
    
    def report_success(self):
        """
        报告成功，智能提速
        
        只有在最近有压力（高负载）的情况下才增速，避免空闲时盲目增加。
        根据当前 in_flight 占比判断应该优先增加哪个：
        - 高占比（>80%）：优先增加 max_concurrent
        - 低占比（<50%）：优先增加 RPS
        """
        self._success_count += 1
        self._success_since_last_limit += 1
        
        # 每 N 次成功，考虑增加速率和并发上限
        if self._success_since_last_limit % self._increase_interval == 0:
            # 检查最近是否有压力，没有压力就不增加
            time_since_pressure = time.monotonic() - self._last_pressure_time
            if time_since_pressure > self._pressure_window:
                # 最近没有高负载，不需要增加容量
                return
            
            old_rps = self._rps
            old_concurrent = self._max_concurrent
            
            # 计算 in_flight 占比
            occupancy = self._in_flight / self._max_concurrent if self._max_concurrent > 0 else 0
            
            if occupancy > 0.8:
                # 经常达到并发上限，优先增加 max_concurrent
                self._max_concurrent += self._concurrent_increase_step * 2
                self._rps += self._increase_step
            elif occupancy < 0.5:
                # 并发数低，优先增加 RPS
                self._rps += self._increase_step * 2
                self._max_concurrent += self._concurrent_increase_step
            else:
                # 均衡增加
                self._rps += self._increase_step
                self._max_concurrent += self._concurrent_increase_step
            
            # 每跨越 100 的倍数时打印日志
            if int(old_rps / 100) != int(self._rps / 100):
                print(f"[RateLimiter] Increased: RPS {old_rps:.1f} -> {self._rps:.1f}, "
                      f"max_concurrent: {old_concurrent} -> {self._max_concurrent} "
                      f"(in-flight: {self._in_flight}, occupancy={occupancy:.0%})")
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
    
    @property
    def in_flight(self):
        """当前正在执行的任务数"""
        return self._in_flight
    
    def get_stats(self):
        return {
            "current_rps": self._rps,
            "max_concurrent": self._max_concurrent,
            "in_flight": self._in_flight,
            "peak_in_flight": self._peak_in_flight,
            "rate_limit_count": self._rate_limit_count,
            "success_count": self._success_count,
        }


# ============================================================================
# 全局配置与状态
# ============================================================================

INITIAL_RPS = 1000        # 初始速率（激进起步）
MIN_RPS = 10              # 最小速率
MAX_CONCURRENT = 800      # 初始最大并发数（激进起步）
MIN_CONCURRENT = 50       # 最小并发数（降速下限）
INCREASE_STEP = 10.0      # 每次增加的 RPS
CONCURRENT_INCREASE_STEP = 20  # 每次增加的并发数
INCREASE_INTERVAL = 20    # 每多少次成功后增加
DECREASE_FACTOR = 0.5     # 遇到 rate limit 时乘以的系数（备用）
MAX_RETRIES = 3
LOG_TOKEN = True

_global_rate_limiter = None
_client_cache = {}

model_name = None
total_prompt_tokens, total_completion_tokens, call_count = 0, 0, 0
current_prompt_tokens, current_completion_tokens = 0, 0


def load_api_configs():
    """Load API configurations from yaml file"""
    config_path = os.path.join(os.path.dirname(__file__), 'apikey.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


# Load configurations at module import time
config = load_api_configs()
api_configs = {k: v for k, v in config.items() if k != 'model2base'}
model2base = config.get('model2base', {})


# ============================================================================
# 辅助函数
# ============================================================================

def _get_client(base_url: str, api_key: str) -> openai.AsyncClient:
    """获取或创建缓存的客户端"""
    key = (base_url, api_key)
    if key not in _client_cache:
        _client_cache[key] = openai.AsyncClient(base_url=base_url, api_key=api_key)
    return _client_cache[key]


def _parse_response(response, is_chat: bool = True):
    """解析响应，返回 (content, usage)"""
    if isinstance(response, str):
        return response, None
    if is_chat:
        return response.choices[0].message.content, getattr(response, 'usage', None)
    return response.choices[0].text, getattr(response, 'usage', None)


def _update_tokens(usage, log_token: bool):
    """更新 token 统计"""
    global current_prompt_tokens, current_completion_tokens
    if LOG_TOKEN and log_token and usage:
        current_prompt_tokens = usage.prompt_tokens
        current_completion_tokens = usage.completion_tokens
        update_token()


def _get_base_and_client(model: str):
    """根据模型获取 base 配置和客户端"""
    if model in model2base:
        base = model2base[model]
    else:
        base = model2base.get("default", "default")
    if base not in api_configs:
        base = "default"
    
    base_config = api_configs[base]
    api_key = random.choice(base_config["api_key"])
    client = _get_client(base_config["url"], api_key)
    return client


# ============================================================================
# 速率限制器管理
# ============================================================================

def set_rate_limit(initial_rps: float = None, min_rps: float = None,
                   max_concurrent: int = None, min_concurrent: int = None,
                   increase_step: float = None, concurrent_increase_step: int = None,
                   increase_interval: int = None, decrease_factor: float = None):
    """配置速率限制参数"""
    global INITIAL_RPS, MIN_RPS, MAX_CONCURRENT, MIN_CONCURRENT, INCREASE_STEP, CONCURRENT_INCREASE_STEP, INCREASE_INTERVAL, DECREASE_FACTOR, _global_rate_limiter
    if initial_rps is not None:
        INITIAL_RPS = initial_rps
    if min_rps is not None:
        MIN_RPS = min_rps
    if max_concurrent is not None:
        MAX_CONCURRENT = max_concurrent
    if min_concurrent is not None:
        MIN_CONCURRENT = min_concurrent
    if increase_step is not None:
        INCREASE_STEP = increase_step
    if concurrent_increase_step is not None:
        CONCURRENT_INCREASE_STEP = concurrent_increase_step
    if increase_interval is not None:
        INCREASE_INTERVAL = increase_interval
    if decrease_factor is not None:
        DECREASE_FACTOR = decrease_factor
    _global_rate_limiter = None  # 重置，下次调用时使用新配置


async def _get_rate_limiter() -> AdaptiveRateLimiter:
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter(
            initial_rps=INITIAL_RPS,
            min_rps=MIN_RPS,
            max_concurrent=MAX_CONCURRENT,
            min_concurrent=MIN_CONCURRENT,
            increase_step=INCREASE_STEP,
            concurrent_increase_step=CONCURRENT_INCREASE_STEP,
            increase_interval=INCREASE_INTERVAL,
            decrease_factor=DECREASE_FACTOR,
        )
    return _global_rate_limiter


def get_rate_limiter_stats():
    """获取当前速率限制器统计信息"""
    if _global_rate_limiter is not None:
        return _global_rate_limiter.get_stats()
    return None


def reset_rate_limiter():
    """重置速率限制器（重新开始探测）"""
    global _global_rate_limiter
    _global_rate_limiter = None


# ============================================================================
# 主要 API 函数
# ============================================================================

def set_model(model):
    global model_name
    model_name = model
    

def set_log_token(log):
    global LOG_TOKEN
    LOG_TOKEN = log


async def gen(prompt=None, messages=None, model="gpt-4o-mini", temperature=1.0, 
              response_format="text", log_token=True, use_template=True, max_tokens=8192*2,
              stop=None, cross_process_limit: int = None):
    """
    生成 LLM 响应
    
    Args:
        cross_process_limit: 跨进程并发限制。如果设置，会等待全局并发数低于此值才开始调用。
                            这个限制是跨所有 Python 进程的！
    """
    rate_limiter = await _get_rate_limiter()
    monitor = get_llm_monitor()
    
    # 如果设置了跨进程限制，先等待
    if cross_process_limit is not None:
        await monitor.wait_if_too_many_async(cross_process_limit)
    
    # 注册调用
    call_id = monitor.register_call(model=model)
    
    try:
        async with rate_limiter:
            return await _gen_impl(prompt, messages, model, temperature, response_format, 
                                   log_token, use_template, max_tokens, rate_limiter, stop)
    finally:
        # 确保调用结束后注销
        monitor.unregister_call(call_id)


async def _gen_impl(prompt=None, messages=None, model="gpt-4o-mini", temperature=1.0, 
                    response_format="text", log_token=True, use_template=True, 
                    max_tokens=8192*2, rate_limiter=None, stop=None):
    """Core generation logic"""
    global call_count, model_name
    
    if not model:
        model = model_name
    
    client = _get_base_and_client(model)
    errors = []
    retry_base = random.uniform(0.1, 2)
    
    if LOG_TOKEN:
        call_count += 1

    # Text completion mode
    if not use_template:
        if not prompt:
            raise ValueError("Prompt must be provided when use_template=False")
        
        for retry in range(MAX_RETRIES):
            try:
                async with asyncio.timeout(360):
                    content, usage = await _try_completion(client, model, prompt, 
                                                           temperature, max_tokens, response_format)
                    _update_tokens(usage, log_token)
                    if rate_limiter:
                        rate_limiter.report_success()
                    return content, []
                    
            except (asyncio.TimeoutError, openai.RateLimitError, openai.APIError, Exception) as e:
                errors.append(_handle_error(e, rate_limiter))
                print(_format_error_log(errors, retry, MAX_RETRIES))
                await asyncio.sleep(retry_base * (2 ** retry))

        print(_format_error_log(errors, MAX_RETRIES, MAX_RETRIES, is_final=True))
        return None, []
    
    # Chat completion mode
    if not messages:
        if not prompt:
            raise ValueError("Either prompt or messages must be provided")
        messages = [{"role": "user", "content": prompt}]
    elif prompt:
        messages.append({"role": "user", "content": prompt})

    for retry in range(MAX_RETRIES):
        try:
            async with asyncio.timeout(240):
                response = await _chat_completion(client, model, messages, 
                                                   temperature, max_tokens, response_format, stop)
                content, usage = _parse_response(response, is_chat=True)
                _update_tokens(usage, log_token)
                
                if rate_limiter:
                    rate_limiter.report_success()
                
                messages.append({"role": "assistant", "content": content})
                return content, messages
                
        except (asyncio.TimeoutError, openai.RateLimitError, openai.APIError, Exception) as e:
            errors.append(_handle_error(e, rate_limiter))
            print(_format_error_log(errors, retry, MAX_RETRIES))
            await asyncio.sleep(retry_base * (2 ** retry))

    print(_format_error_log(errors, MAX_RETRIES, MAX_RETRIES, is_final=True))
    return None, messages


async def _try_completion(client, model, prompt, temperature, max_tokens, response_format):
    """尝试 completion API，失败则回退到 chat API"""
    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None
        )
        return _parse_response(response, is_chat=False)
    except openai.APIError as e:
        if "OperationNotSupported" in str(e) or "completion operation does not work" in str(e):
            # Fallback to chat API
            messages = [
                {"role": "system", "content": "You are a text completion assistant. Continue the given text naturally without adding any introduction, explanation, or conversation. Just directly continue where the text left off."},
                {"role": "user", "content": f"Continue this text:\n\n{prompt}"}
            ]
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": response_format}
            )
            return _parse_response(response, is_chat=True)
        raise


async def _chat_completion(client, model, messages, temperature, max_tokens, response_format, stop=None):
    """Execute chat completion"""
    # Some models don't support max_tokens parameter
    if model in ["o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stop=stop,
            response_format={"type": response_format}
        )
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop,
        max_tokens=max_tokens,
        response_format={"type": response_format}
    )


def _handle_error(e, rate_limiter) -> str:
    """处理错误并返回错误描述"""
    if isinstance(e, asyncio.TimeoutError):
        return "Timeout"
    if isinstance(e, openai.RateLimitError):
        if rate_limiter:
            rate_limiter.report_rate_limit()
        return "RateLimit"
    if isinstance(e, openai.APIError):
        # 提取关键错误信息，去除冗余内容
        err_str = str(e)
        if len(err_str) > 100:
            err_str = err_str[:100] + "..."
        return f"API({err_str})"
    return f"{type(e).__name__}"


def _format_error_log(errors: list, retry: int, max_retries: int, is_final: bool = False) -> str:
    """格式化错误日志输出"""
    if is_final:
        return f"[LLM] ✗ Failed after {max_retries} retries | Errors: {' → '.join(errors)}"
    else:
        return f"[LLM] Retry {retry + 1}/{max_retries} | {errors[-1] if errors else 'Unknown'}"


# ============================================================================
# Token 统计函数
# ============================================================================

def update_token():
    global total_prompt_tokens, total_completion_tokens
    total_prompt_tokens += current_prompt_tokens
    total_completion_tokens += current_completion_tokens


def reset_token():
    global total_prompt_tokens, total_completion_tokens, call_count
    total_prompt_tokens = 0
    total_completion_tokens = 0
    call_count = 0


def get_model():
    return model_name


def get_token():
    return total_prompt_tokens, total_completion_tokens


def get_call_count():
    return call_count


def get_current_tokens():
    return current_prompt_tokens, current_completion_tokens


def reset_current_tokens():
    global current_prompt_tokens, current_completion_tokens
    current_prompt_tokens = 0
    current_completion_tokens = 0


# ============================================================================
# 跨进程监控便捷函数
# ============================================================================

def get_system_llm_count() -> int:
    """获取系统中当前正在进行的 LLM 调用总数（跨所有进程）"""
    return get_llm_monitor().get_active_count()


def get_system_llm_status() -> dict:
    """获取系统中 LLM 调用的详细状态"""
    return get_llm_monitor().get_status()


def set_cross_process_limit(max_concurrent: int):
    """
    设置跨进程并发限制
    
    调用此函数后，所有通过 gen() 发起的调用都会遵守这个全局限制
    """
    global CROSS_PROCESS_LIMIT
    CROSS_PROCESS_LIMIT = max_concurrent


# 全局跨进程限制（None 表示不限制）
CROSS_PROCESS_LIMIT = None


# ============================================================================
# 命令行工具：监控系统中的 LLM 调用
# ============================================================================

def _cli_monitor():
    """命令行监控工具入口"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="LLM 调用跨进程监控工具")
    parser.add_argument("--watch", "-w", action="store_true", 
                        help="持续监控模式，每秒刷新")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                        help="监控刷新间隔（秒）")
    parser.add_argument("--json", "-j", action="store_true",
                        help="以 JSON 格式输出")
    
    args = parser.parse_args()
    
    monitor = get_llm_monitor()
    
    if args.watch:
        # 持续监控模式
        try:
            while True:
                if not args.json:
                    # 清屏
                    print("\033[2J\033[H", end="")
                    print_llm_status()
                else:
                    status = monitor.get_status()
                    print(json.dumps(status))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控结束")
    else:
        # 单次输出
        if args.json:
            status = monitor.get_status()
            print(json.dumps(status, indent=2))
        else:
            print_llm_status()


if __name__ == "__main__":
    _cli_monitor()
