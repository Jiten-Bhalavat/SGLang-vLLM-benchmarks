#!/usr/bin/env python3
"""
Results Parser and Analyzer
Aggregates benchmark results and generates analysis reports

Author: Cloud LLM Benchmark Project
Course: MSML 650 - Cloud Computing
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class ResultsAnalyzer:
    """Analyzes and aggregates benchmark results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.vllm_results: List[Dict] = []
        self.sglang_results: List[Dict] = []
        
    def load_results(self) -> None:
        """Load all result files from the results directory"""
        
        # Load vLLM results
        vllm_dir = self.results_dir / "vllm"
        if vllm_dir.exists():
            for result_file in vllm_dir.glob("*.json"):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = str(result_file)
                    self.vllm_results.append(data)
        
        # Load SGLang results
        sglang_dir = self.results_dir / "sglang"
        if sglang_dir.exists():
            for result_file in sglang_dir.glob("*.json"):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = str(result_file)
                    self.sglang_results.append(data)
        
        console.print(f"[green]Loaded {len(self.vllm_results)} vLLM results[/green]")
        console.print(f"[green]Loaded {len(self.sglang_results)} SGLang results[/green]")
    
    def get_summary_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for a list of results"""
        
        if not results:
            return {}
        
        throughputs = [r.get('throughput_tps', 0) for r in results if r.get('throughput_tps', 0) > 0]
        latencies = [r.get('avg_latency', 0) for r in results if r.get('avg_latency', 0) > 0]
        p95_latencies = [r.get('p95_latency', 0) for r in results if r.get('p95_latency', 0) > 0]
        
        success_rates = []
        for r in results:
            total = r.get('total_requests', 0)
            success = r.get('successful_requests', 0)
            if total > 0:
                success_rates.append(success / total * 100)
        
        return {
            'num_benchmarks': len(results),
            'avg_throughput_tps': statistics.mean(throughputs) if throughputs else 0,
            'max_throughput_tps': max(throughputs) if throughputs else 0,
            'min_throughput_tps': min(throughputs) if throughputs else 0,
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'avg_p95_latency': statistics.mean(p95_latencies) if p95_latencies else 0,
            'avg_success_rate': statistics.mean(success_rates) if success_rates else 0
        }
    
    def compare_frameworks(self) -> Dict[str, Any]:
        """Generate comparison between vLLM and SGLang"""
        
        vllm_stats = self.get_summary_stats(self.vllm_results)
        sglang_stats = self.get_summary_stats(self.sglang_results)
        
        comparison = {
            'vllm': vllm_stats,
            'sglang': sglang_stats,
            'comparison': {}
        }
        
        if vllm_stats and sglang_stats:
            vllm_tps = vllm_stats.get('avg_throughput_tps', 0)
            sglang_tps = sglang_stats.get('avg_throughput_tps', 0)
            
            vllm_lat = vllm_stats.get('avg_latency', 0)
            sglang_lat = sglang_stats.get('avg_latency', 0)
            
            comparison['comparison'] = {
                'throughput_winner': 'vllm' if vllm_tps > sglang_tps else 'sglang',
                'throughput_diff_percent': ((vllm_tps - sglang_tps) / sglang_tps * 100) if sglang_tps > 0 else 0,
                'latency_winner': 'vllm' if vllm_lat < sglang_lat else 'sglang',
                'latency_diff_percent': ((sglang_lat - vllm_lat) / sglang_lat * 100) if sglang_lat > 0 else 0
            }
        
        return comparison
    
    def print_summary(self) -> None:
        """Print a formatted summary of all results"""
        
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]Benchmark Results Summary[/bold blue]",
            title="📊 Analysis Report"
        ))
        
        # vLLM Summary
        if self.vllm_results:
            console.print("\n[bold cyan]vLLM Results[/bold cyan]")
            vllm_stats = self.get_summary_stats(self.vllm_results)
            self._print_stats_table("vLLM", vllm_stats)
        
        # SGLang Summary
        if self.sglang_results:
            console.print("\n[bold cyan]SGLang Results[/bold cyan]")
            sglang_stats = self.get_summary_stats(self.sglang_results)
            self._print_stats_table("SGLang", sglang_stats)
        
        # Comparison
        if self.vllm_results and self.sglang_results:
            comparison = self.compare_frameworks()
            self._print_comparison(comparison)
    
    def _print_stats_table(self, framework: str, stats: Dict) -> None:
        """Print statistics in a formatted table"""
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Number of Benchmarks", str(stats.get('num_benchmarks', 0)))
        table.add_row("Avg Throughput (TPS)", f"{stats.get('avg_throughput_tps', 0):.2f}")
        table.add_row("Max Throughput (TPS)", f"{stats.get('max_throughput_tps', 0):.2f}")
        table.add_row("Min Throughput (TPS)", f"{stats.get('min_throughput_tps', 0):.2f}")
        table.add_row("Avg Latency (s)", f"{stats.get('avg_latency', 0):.3f}")
        table.add_row("Avg P95 Latency (s)", f"{stats.get('avg_p95_latency', 0):.3f}")
        table.add_row("Avg Success Rate", f"{stats.get('avg_success_rate', 0):.1f}%")
        
        console.print(table)
    
    def _print_comparison(self, comparison: Dict) -> None:
        """Print framework comparison"""
        
        console.print("\n[bold yellow]Framework Comparison[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("vLLM", justify="right", style="green")
        table.add_column("SGLang", justify="right", style="blue")
        table.add_column("Winner", justify="center")
        
        vllm = comparison.get('vllm', {})
        sglang = comparison.get('sglang', {})
        comp = comparison.get('comparison', {})
        
        # Throughput comparison
        throughput_winner = comp.get('throughput_winner', 'N/A')
        winner_style = "[bold green]" if throughput_winner == "vllm" else "[bold blue]"
        table.add_row(
            "Avg Throughput (TPS)",
            f"{vllm.get('avg_throughput_tps', 0):.2f}",
            f"{sglang.get('avg_throughput_tps', 0):.2f}",
            f"{winner_style}{throughput_winner.upper()}[/]"
        )
        
        # Latency comparison
        latency_winner = comp.get('latency_winner', 'N/A')
        winner_style = "[bold green]" if latency_winner == "vllm" else "[bold blue]"
        table.add_row(
            "Avg Latency (s)",
            f"{vllm.get('avg_latency', 0):.3f}",
            f"{sglang.get('avg_latency', 0):.3f}",
            f"{winner_style}{latency_winner.upper()}[/]"
        )
        
        # P95 Latency
        p95_winner = "vllm" if vllm.get('avg_p95_latency', float('inf')) < sglang.get('avg_p95_latency', float('inf')) else "sglang"
        winner_style = "[bold green]" if p95_winner == "vllm" else "[bold blue]"
        table.add_row(
            "Avg P95 Latency (s)",
            f"{vllm.get('avg_p95_latency', 0):.3f}",
            f"{sglang.get('avg_p95_latency', 0):.3f}",
            f"{winner_style}{p95_winner.upper()}[/]"
        )
        
        console.print(table)
        
        # Print summary
        console.print("\n[bold]Summary:[/bold]")
        if comp:
            tps_diff = comp.get('throughput_diff_percent', 0)
            if tps_diff > 0:
                console.print(f"  • vLLM is [green]{abs(tps_diff):.1f}%[/green] faster in throughput")
            else:
                console.print(f"  • SGLang is [blue]{abs(tps_diff):.1f}%[/blue] faster in throughput")
            
            lat_diff = comp.get('latency_diff_percent', 0)
            if lat_diff > 0:
                console.print(f"  • vLLM has [green]{abs(lat_diff):.1f}%[/green] lower latency")
            else:
                console.print(f"  • SGLang has [blue]{abs(lat_diff):.1f}%[/blue] lower latency")
    
    def export_to_csv(self, output_path: str = "results/benchmark_summary.csv") -> None:
        """Export results to CSV for further analysis"""
        
        all_results = []
        
        for result in self.vllm_results:
            all_results.append({
                'framework': 'vllm',
                'workload': result.get('workload_name', 'unknown'),
                'throughput_tps': result.get('throughput_tps', 0),
                'throughput_rps': result.get('throughput_rps', 0),
                'avg_latency': result.get('avg_latency', 0),
                'min_latency': result.get('min_latency', 0),
                'max_latency': result.get('max_latency', 0),
                'p50_latency': result.get('p50_latency', 0),
                'p90_latency': result.get('p90_latency', 0),
                'p95_latency': result.get('p95_latency', 0),
                'p99_latency': result.get('p99_latency', 0),
                'total_requests': result.get('total_requests', 0),
                'successful_requests': result.get('successful_requests', 0),
                'failed_requests': result.get('failed_requests', 0),
                'total_time': result.get('total_time', 0),
                'timestamp': result.get('timestamp', '')
            })
        
        for result in self.sglang_results:
            all_results.append({
                'framework': 'sglang',
                'workload': result.get('workload_name', 'unknown'),
                'throughput_tps': result.get('throughput_tps', 0),
                'throughput_rps': result.get('throughput_rps', 0),
                'avg_latency': result.get('avg_latency', 0),
                'min_latency': result.get('min_latency', 0),
                'max_latency': result.get('max_latency', 0),
                'p50_latency': result.get('p50_latency', 0),
                'p90_latency': result.get('p90_latency', 0),
                'p95_latency': result.get('p95_latency', 0),
                'p99_latency': result.get('p99_latency', 0),
                'total_requests': result.get('total_requests', 0),
                'successful_requests': result.get('successful_requests', 0),
                'failed_requests': result.get('failed_requests', 0),
                'total_time': result.get('total_time', 0),
                'timestamp': result.get('timestamp', '')
            })
        
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
            console.print(f"[green]Results exported to: {output_path}[/green]")
        else:
            console.print("[yellow]No results to export[/yellow]")
    
    def generate_report(self, output_path: str = "results/analysis_report.json") -> None:
        """Generate a comprehensive analysis report in JSON format"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_vllm_benchmarks': len(self.vllm_results),
                'total_sglang_benchmarks': len(self.sglang_results)
            },
            'vllm_statistics': self.get_summary_stats(self.vllm_results),
            'sglang_statistics': self.get_summary_stats(self.sglang_results),
            'comparison': self.compare_frameworks(),
            'individual_results': {
                'vllm': self.vllm_results,
                'sglang': self.sglang_results
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"[green]Analysis report saved to: {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Results Parser and Analyzer")
    
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory containing benchmark results"
    )
    
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV"
    )
    
    parser.add_argument(
        "--csv-output",
        type=str,
        default="results/benchmark_summary.csv",
        help="Output path for CSV export"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive analysis report"
    )
    
    parser.add_argument(
        "--report-output",
        type=str,
        default="results/analysis_report.json",
        help="Output path for analysis report"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and load results
    analyzer = ResultsAnalyzer(results_dir=args.results_dir)
    analyzer.load_results()
    
    # Print summary
    analyzer.print_summary()
    
    # Export to CSV if requested
    if args.export_csv:
        analyzer.export_to_csv(args.csv_output)
    
    # Generate report if requested
    if args.generate_report:
        analyzer.generate_report(args.report_output)


if __name__ == "__main__":
    main()

