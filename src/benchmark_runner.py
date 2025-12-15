#!/usr/bin/env python3
"""
Benchmark Runner - Main orchestrator for running benchmarks across frameworks
Supports vLLM and SGLang with configurable workloads

Author: Cloud LLM Benchmark Project
Course: MSML 650 - Cloud Computing
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import time
import requests

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_test import LLMLoadTester, print_results, save_results, BenchmarkResults

console = Console()


class BenchmarkOrchestrator:
    """
    Orchestrates benchmark runs across different frameworks and configurations
    """
    
    # Default server configurations
    FRAMEWORK_CONFIGS = {
        "vllm": {
            "default_port": 8000,
            "health_endpoint": "/health",
            "default_url": "http://localhost:8000"
        },
        "sglang": {
            "default_port": 30000,
            "health_endpoint": "/health",
            "default_url": "http://localhost:30000"
        }
    }
    
    def __init__(
        self,
        results_dir: str = "results",
        model_name: str = "facebook/opt-125m"
    ):
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.all_results: List[BenchmarkResults] = []
        
        # Ensure results directories exist
        (self.results_dir / "vllm").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "sglang").mkdir(parents=True, exist_ok=True)
    
    def check_server_health(self, url: str, framework: str) -> bool:
        """Check if the inference server is healthy and responding"""
        config = self.FRAMEWORK_CONFIGS.get(framework, {})
        health_endpoint = config.get("health_endpoint", "/health")
        
        try:
            response = requests.get(f"{url}{health_endpoint}", timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def wait_for_server(
        self,
        url: str,
        framework: str,
        max_wait: int = 300,
        check_interval: int = 5
    ) -> bool:
        """Wait for server to become healthy"""
        console.print(f"[yellow]Waiting for {framework.upper()} server at {url}...[/yellow]")
        
        start_time = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Waiting for {framework} server...", total=None)
            
            while time.time() - start_time < max_wait:
                if self.check_server_health(url, framework):
                    progress.update(task, description=f"[green]{framework} server is ready![/green]")
                    console.print(f"[green]✓ {framework.upper()} server is healthy![/green]")
                    return True
                
                time.sleep(check_interval)
        
        console.print(f"[red]✗ Timeout waiting for {framework.upper()} server[/red]")
        return False
    
    async def run_single_benchmark(
        self,
        framework: str,
        config_path: str,
        url: Optional[str] = None
    ) -> Optional[BenchmarkResults]:
        """Run a single benchmark with specified configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get URL
        if url is None:
            url = self.FRAMEWORK_CONFIGS[framework]["default_url"]
        
        # Check server health
        if not self.check_server_health(url, framework):
            console.print(f"[red]Error: {framework.upper()} server not responding at {url}[/red]")
            console.print("[yellow]Make sure the server is running before running benchmarks.[/yellow]")
            return None
        
        console.print(Panel(
            f"[bold]Running {framework.upper()} Benchmark[/bold]\n"
            f"Config: {config.get('name', 'unknown')}\n"
            f"Requests: {config.get('num_requests', 10)}\n"
            f"Concurrency: {config.get('concurrency', 5)}",
            title="Benchmark Configuration"
        ))
        
        # Create tester
        tester = LLMLoadTester(
            base_url=url,
            framework=framework,
            model_name=self.model_name,
            timeout=config.get("timeout_seconds", 300)
        )
        
        # Run benchmark
        results = await tester.run_benchmark(
            prompts=config.get("prompts", ["Explain quantum computing."]),
            num_requests=config.get("num_requests", 10),
            max_tokens=config.get("max_tokens", 128),
            concurrency=config.get("concurrency", 5),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            warmup_requests=config.get("warmup_requests", 2)
        )
        
        # Update workload name
        results.workload_name = config.get("name", "benchmark")
        
        # Print results
        print_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / framework / f"{config.get('name', 'benchmark')}_{timestamp}.json"
        save_results(results, str(output_path))
        
        self.all_results.append(results)
        return results
    
    async def run_all_benchmarks(
        self,
        frameworks: List[str],
        config_paths: List[str],
        urls: Optional[Dict[str, str]] = None
    ) -> List[BenchmarkResults]:
        """Run benchmarks across all specified frameworks and configurations"""
        
        results = []
        urls = urls or {}
        
        for framework in frameworks:
            url = urls.get(framework, self.FRAMEWORK_CONFIGS[framework]["default_url"])
            
            for config_path in config_paths:
                console.print(f"\n[bold blue]{'='*60}[/bold blue]")
                console.print(f"[bold]Framework: {framework.upper()} | Config: {config_path}[/bold]")
                console.print(f"[bold blue]{'='*60}[/bold blue]\n")
                
                result = await self.run_single_benchmark(framework, config_path, url)
                if result:
                    results.append(result)
                
                # Add delay between benchmarks
                await asyncio.sleep(2)
        
        return results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate a comparison report of all benchmark results"""
        
        if not self.all_results:
            return {}
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "model": self.model_name,
            "frameworks": {},
            "comparison": {}
        }
        
        # Group results by framework
        for result in self.all_results:
            if result.framework not in report["frameworks"]:
                report["frameworks"][result.framework] = []
            
            report["frameworks"][result.framework].append({
                "workload": result.workload_name,
                "throughput_tps": result.throughput_tps,
                "throughput_rps": result.throughput_rps,
                "avg_latency": result.avg_latency,
                "p50_latency": result.p50_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "success_rate": result.successful_requests / result.total_requests if result.total_requests > 0 else 0
            })
        
        # Calculate comparison metrics
        frameworks = list(report["frameworks"].keys())
        if len(frameworks) == 2:
            fw1, fw2 = frameworks
            fw1_results = report["frameworks"][fw1]
            fw2_results = report["frameworks"][fw2]
            
            if fw1_results and fw2_results:
                avg_tps_1 = sum(r["throughput_tps"] for r in fw1_results) / len(fw1_results)
                avg_tps_2 = sum(r["throughput_tps"] for r in fw2_results) / len(fw2_results)
                
                avg_latency_1 = sum(r["avg_latency"] for r in fw1_results) / len(fw1_results)
                avg_latency_2 = sum(r["avg_latency"] for r in fw2_results) / len(fw2_results)
                
                report["comparison"] = {
                    f"{fw1}_avg_throughput_tps": avg_tps_1,
                    f"{fw2}_avg_throughput_tps": avg_tps_2,
                    "throughput_difference_percent": ((avg_tps_1 - avg_tps_2) / avg_tps_2 * 100) if avg_tps_2 > 0 else 0,
                    f"{fw1}_avg_latency": avg_latency_1,
                    f"{fw2}_avg_latency": avg_latency_2,
                    "latency_difference_percent": ((avg_latency_1 - avg_latency_2) / avg_latency_2 * 100) if avg_latency_2 > 0 else 0
                }
        
        return report
    
    def print_comparison_table(self):
        """Print a comparison table of all results"""
        
        if not self.all_results:
            console.print("[yellow]No results to compare[/yellow]")
            return
        
        table = Table(title="Benchmark Comparison Summary", show_header=True, header_style="bold magenta")
        
        table.add_column("Framework", style="cyan")
        table.add_column("Workload", style="white")
        table.add_column("Throughput (TPS)", justify="right", style="green")
        table.add_column("Avg Latency (s)", justify="right", style="yellow")
        table.add_column("P95 Latency (s)", justify="right", style="yellow")
        table.add_column("Success Rate", justify="right", style="blue")
        
        for result in self.all_results:
            success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
            
            table.add_row(
                result.framework.upper(),
                result.workload_name,
                f"{result.throughput_tps:.2f}",
                f"{result.avg_latency:.3f}",
                f"{result.p95_latency:.3f}",
                f"{success_rate:.1f}%"
            )
        
        console.print("\n")
        console.print(table)
        console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run vLLM benchmark with medium workload
  python benchmark_runner.py --framework vllm --config configs/workload_medium.json
  
  # Run SGLang benchmark with custom URL
  python benchmark_runner.py --framework sglang --config configs/workload_small.json --url http://localhost:30000
  
  # Run all benchmarks (both frameworks, all workloads)
  python benchmark_runner.py --all --config-dir configs/
        """
    )
    
    parser.add_argument(
        "--framework", "-f",
        type=str,
        choices=["vllm", "sglang", "both"],
        default="vllm",
        help="Framework to benchmark (default: vllm)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to workload configuration JSON file"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing workload configuration files"
    )
    
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="Server URL (overrides default)"
    )
    
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang server URL"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="facebook/opt-125m",
        help="Model name (default: facebook/opt-125m)"
    )
    
    parser.add_argument(
        "--results-dir", "-o",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all configurations in config directory"
    )
    
    parser.add_argument(
        "--wait-for-server",
        action="store_true",
        help="Wait for server to become healthy before running"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(
        results_dir=args.results_dir,
        model_name=args.model
    )
    
    # Determine frameworks to run
    if args.framework == "both":
        frameworks = ["vllm", "sglang"]
    else:
        frameworks = [args.framework]
    
    # Determine configs to run
    config_paths = []
    if args.config:
        config_paths = [args.config]
    elif args.all:
        config_dir = Path(args.config_dir)
        config_paths = sorted([str(p) for p in config_dir.glob("workload_*.json")])
    else:
        # Default to medium workload
        default_config = Path(args.config_dir) / "workload_medium.json"
        if default_config.exists():
            config_paths = [str(default_config)]
        else:
            console.print("[red]No configuration file specified. Use --config or --all[/red]")
            sys.exit(1)
    
    if not config_paths:
        console.print("[red]No configuration files found[/red]")
        sys.exit(1)
    
    # Set up URLs
    urls = {
        "vllm": args.url or args.vllm_url,
        "sglang": args.url or args.sglang_url
    }
    
    # Print banner
    console.print(Panel.fit(
        "[bold blue]Cloud LLM Inference Benchmark[/bold blue]\n"
        f"Frameworks: {', '.join(f.upper() for f in frameworks)}\n"
        f"Configs: {len(config_paths)} workload(s)\n"
        f"Model: {args.model}",
        title="🚀 Benchmark Runner"
    ))
    
    # Wait for server if requested
    if args.wait_for_server:
        for framework in frameworks:
            url = urls[framework]
            if not orchestrator.wait_for_server(url, framework):
                console.print(f"[red]Cannot connect to {framework} server. Exiting.[/red]")
                sys.exit(1)
    
    # Run benchmarks
    async def run():
        await orchestrator.run_all_benchmarks(frameworks, config_paths, urls)
        
        # Print comparison
        orchestrator.print_comparison_table()
        
        # Generate and save comparison report
        report = orchestrator.generate_comparison_report()
        if report:
            report_path = Path(args.results_dir) / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            console.print(f"[green]Comparison report saved to: {report_path}[/green]")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()

