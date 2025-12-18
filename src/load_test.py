#!/usr/bin/env python3
"""
Load Testing Module for LLM Inference Benchmarking
Supports both vLLM and SGLang servers with OpenAI-compatible APIs

Author: Cloud LLM Benchmark Project
Course: MSML 650 - Cloud Computing
"""

import asyncio
import aiohttp
import time
import json
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class RequestResult:
    """Result of a single inference request"""
    request_id: int
    prompt: str
    success: bool
    latency: float  # Total time in seconds
    time_to_first_token: Optional[float] = None
    output_tokens: int = 0
    input_tokens: int = 0
    tokens_per_second: float = 0.0
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    framework: str
    model_name: str
    workload_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    avg_ttft: Optional[float]
    throughput_rps: float  # Requests per second
    throughput_tps: float  # Tokens per second
    total_output_tokens: int
    total_input_tokens: int
    individual_results: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LLMLoadTester:
    """Async load tester for LLM inference servers"""
    
    def __init__(
        self,
        base_url: str,
        framework: str = "vllm",
        model_name: str = "unknown",
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.framework = framework.lower()
        self.model_name = model_name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Set endpoint based on framework
        if self.framework == "vllm":
            self.endpoint = f"{self.base_url}/v1/completions"
        elif self.framework == "sglang":
            self.endpoint = f"{self.base_url}/generate"
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    async def _make_request_vllm(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict:
        """Make request to vLLM server (OpenAI-compatible API)"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        async with session.post(self.endpoint, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "text": data["choices"][0]["text"],
                    "usage": data.get("usage", {})
                }
            else:
                error_text = await response.text()
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}"
                }
    
    async def _make_request_sglang(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict:
        """Make request to SGLang server"""
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        async with session.post(self.endpoint, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "text": data.get("text", ""),
                    "usage": {
                        "completion_tokens": data.get("meta_info", {}).get("completion_tokens", 0),
                        "prompt_tokens": data.get("meta_info", {}).get("prompt_tokens", 0)
                    }
                }
            else:
                error_text = await response.text()
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}"
                }
    
    async def single_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> RequestResult:
        """Execute a single inference request and measure performance"""
        
        start_time = time.perf_counter()
        
        try:
            if self.framework == "vllm":
                result = await self._make_request_vllm(
                    session, prompt, max_tokens, temperature, top_p
                )
            else:
                result = await self._make_request_sglang(
                    session, prompt, max_tokens, temperature, top_p
                )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            if result["success"]:
                output_text = result["text"]
                usage = result.get("usage", {})
                
                # Estimate tokens if not provided
                output_tokens = usage.get("completion_tokens", len(output_text.split()))
                input_tokens = usage.get("prompt_tokens", len(prompt.split()))
                
                tokens_per_second = output_tokens / latency if latency > 0 else 0
                
                return RequestResult(
                    request_id=request_id,
                    prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    success=True,
                    latency=latency,
                    output_tokens=output_tokens,
                    input_tokens=input_tokens,
                    tokens_per_second=tokens_per_second
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    success=False,
                    latency=latency,
                    error_message=result.get("error", "Unknown error")
                )
                
        except asyncio.TimeoutError:
            return RequestResult(
                request_id=request_id,
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                success=False,
                latency=self.timeout.total,
                error_message="Request timeout"
            )
        except Exception as e:
            end_time = time.perf_counter()
            return RequestResult(
                request_id=request_id,
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                success=False,
                latency=end_time - start_time,
                error_message=str(e)
            )
    
    async def run_benchmark(
        self,
        prompts: List[str],
        num_requests: int,
        max_tokens: int = 128,
        concurrency: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        warmup_requests: int = 2
    ) -> BenchmarkResults:
        """Run benchmark with specified workload"""
        
        console.print(f"\n[bold blue]Starting benchmark for {self.framework.upper()}[/bold blue]")
        console.print(f"  Model: {self.model_name}")
        console.print(f"  Endpoint: {self.endpoint}")
        console.print(f"  Requests: {num_requests}, Concurrency: {concurrency}")
        console.print(f"  Max tokens: {max_tokens}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session, req_id, prompt):
            async with semaphore:
                return await self.single_request(
                    session, req_id, prompt, max_tokens, temperature, top_p
                )
        
        results: List[RequestResult] = []
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Warmup phase
            if warmup_requests > 0:
                console.print(f"\n[yellow]Running {warmup_requests} warmup requests...[/yellow]")
                warmup_tasks = [
                    bounded_request(session, -i, prompts[i % len(prompts)])
                    for i in range(warmup_requests)
                ]
                await asyncio.gather(*warmup_tasks)
                console.print("[green]Warmup complete![/green]")
            
            # Main benchmark
            console.print(f"\n[bold]Running {num_requests} benchmark requests...[/bold]")
            
            start_time = time.perf_counter()
            
            tasks = [
                bounded_request(session, i, prompts[i % len(prompts)])
                for i in range(num_requests)
            ]
            
            # Run with progress bar
            with tqdm(total=num_requests, desc="Benchmarking") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
            
            total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        return self._calculate_results(results, total_time)
    
    def _calculate_results(
        self,
        results: List[RequestResult],
        total_time: float
    ) -> BenchmarkResults:
        """Calculate aggregated benchmark statistics"""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if not successful:
            console.print("[red]All requests failed![/red]")
            return BenchmarkResults(
                framework=self.framework,
                model_name=self.model_name,
                workload_name="benchmark",
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(results),
                total_time=total_time,
                avg_latency=0,
                min_latency=0,
                max_latency=0,
                p50_latency=0,
                p90_latency=0,
                p95_latency=0,
                p99_latency=0,
                avg_ttft=None,
                throughput_rps=0,
                throughput_tps=0,
                total_output_tokens=0,
                total_input_tokens=0,
                individual_results=[asdict(r) for r in results]
            )
        
        latencies = [r.latency for r in successful]
        tps_values = [r.tokens_per_second for r in successful]
        
        total_output_tokens = sum(r.output_tokens for r in successful)
        total_input_tokens = sum(r.input_tokens for r in successful)
        
        return BenchmarkResults(
            framework=self.framework,
            model_name=self.model_name,
            workload_name="benchmark",
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time=total_time,
            avg_latency=statistics.mean(latencies),
            min_latency=min(latencies),
            max_latency=max(latencies),
            p50_latency=np.percentile(latencies, 50),
            p90_latency=np.percentile(latencies, 90),
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            avg_ttft=None,  # Would need streaming to measure
            throughput_rps=len(successful) / total_time,
            throughput_tps=total_output_tokens / total_time,
            total_output_tokens=total_output_tokens,
            total_input_tokens=total_input_tokens,
            individual_results=[asdict(r) for r in results]
        )


def print_results(results: BenchmarkResults):
    """Pretty print benchmark results"""
    
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]BENCHMARK RESULTS - {results.framework.upper()}[/bold green]")
    console.print("=" * 60)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Framework", results.framework.upper())
    table.add_row("Model", results.model_name)
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Total Requests", str(results.total_requests))
    table.add_row("Successful", f"[green]{results.successful_requests}[/green]")
    table.add_row("Failed", f"[red]{results.failed_requests}[/red]")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Total Time", f"{results.total_time:.2f} s")
    table.add_row("Throughput (RPS)", f"{results.throughput_rps:.2f} req/s")
    table.add_row("Throughput (TPS)", f"[bold]{results.throughput_tps:.2f} tok/s[/bold]")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Avg Latency", f"{results.avg_latency:.3f} s")
    table.add_row("Min Latency", f"{results.min_latency:.3f} s")
    table.add_row("Max Latency", f"{results.max_latency:.3f} s")
    table.add_row("P50 Latency", f"{results.p50_latency:.3f} s")
    table.add_row("P90 Latency", f"{results.p90_latency:.3f} s")
    table.add_row("P95 Latency", f"{results.p95_latency:.3f} s")
    table.add_row("P99 Latency", f"{results.p99_latency:.3f} s")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Total Output Tokens", str(results.total_output_tokens))
    table.add_row("Total Input Tokens", str(results.total_input_tokens))
    
    console.print(table)
    console.print("=" * 60 + "\n")


def save_results(results: BenchmarkResults, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    console.print(f"[green]Results saved to: {output_path}[/green]")


async def main():
    parser = argparse.ArgumentParser(description="LLM Inference Load Tester")
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the inference server"
    )
    parser.add_argument(
        "--framework", "-f",
        type=str,
        choices=["vllm", "sglang"],
        default="vllm",
        help="Framework type (vllm or sglang)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="facebook/opt-125m",
        help="Model name for API requests"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to workload configuration JSON file"
    )
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=10,
        help="Number of requests to send"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--concurrency", "-p",
        type=int,
        default=5,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results JSON"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Prompt to use for testing"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    prompts = [args.prompt]
    num_requests = args.num_requests
    max_tokens = args.max_tokens
    concurrency = args.concurrency
    warmup_requests = 2
    temperature = 0.7
    top_p = 0.95
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            prompts = config.get("prompts", prompts)
            num_requests = config.get("num_requests", num_requests)
            max_tokens = config.get("max_tokens", max_tokens)
            concurrency = config.get("concurrency", concurrency)
            warmup_requests = config.get("warmup_requests", warmup_requests)
            temperature = config.get("temperature", temperature)
            top_p = config.get("top_p", top_p)
    
    # Create tester and run benchmark
    tester = LLMLoadTester(
        base_url=args.url,
        framework=args.framework,
        model_name=args.model,
        timeout=300
    )
    
    results = await tester.run_benchmark(
        prompts=prompts,
        num_requests=num_requests,
        max_tokens=max_tokens,
        concurrency=concurrency,
        temperature=temperature,
        top_p=top_p,
        warmup_requests=warmup_requests
    )
    
    # Print results
    print_results(results)
    
    # Save results if output path provided
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{args.framework}/benchmark_{timestamp}.json"
        save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())



