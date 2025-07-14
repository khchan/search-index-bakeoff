#!/usr/bin/env python3
"""
Performance benchmarking script for the search-index-reader service.

This script performs comprehensive performance testing including:
- Memory usage monitoring
- Concurrent request load testing  
- Latency measurement and breakdown
- Breaking point detection
- Performance visualization and reporting
"""

import asyncio
import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import click
import psutil
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent))

from clients.reader_client import AsyncSearchIndexReaderClient

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tests."""
    tenant_id: str = "1466591000091951104"
    table_name: str = "hierarchies_1466593582621392896"
    reader_url: str = "http://localhost:8002"
    auth_user: Optional[str] = None
    auth_key: Optional[str] = None
    output_dir: str = "benchmark_results"
    max_concurrent: int = 100
    test_duration: int = 300  # seconds
    request_timeout: float = 30.0

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: float
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    concurrent_requests: int = 1
    query: str = ""
    is_timeout: bool = False
    is_throttled: bool = False
    retrieval_time_ms: Optional[float] = None  # Time for just retrieval
    total_time_ms: Optional[float] = None      # End-to-end time

@dataclass
class WorkerMetrics:
    """Metrics for a single worker process."""
    pid: int
    memory_mb: float

@dataclass
class SystemMetrics:
    """System resource metrics including individual worker data."""
    timestamp: float
    total_memory_mb: float
    concurrent_requests: int = 0
    worker_metrics: List[WorkerMetrics] = None
    
    def __post_init__(self):
        if self.worker_metrics is None:
            self.worker_metrics = []

@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config: BenchmarkConfig
    request_metrics: List[RequestMetrics]
    system_metrics: List[SystemMetrics]
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float
    max_memory_mb: float
    breaking_point_concurrent: Optional[int] = None
    total_qps: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    throttle_rate: float = 0.0
    http_4xx_rate: float = 0.0
    http_5xx_rate: float = 0.0
    
CONCURRENCY_LEVELS = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]

class ServiceMonitor:
    """Monitor reader service Gunicorn worker processes for memory and CPU usage."""
    
    def __init__(self, service_port: int = 8002):
        self.service_port = service_port
        self.worker_procs = []
        self._find_worker_processes()
    
    def _find_worker_processes(self):
        """Find all Gunicorn worker processes for the reader service by port and command line."""
        self.worker_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
            try:
                cmdline = proc.info.get('cmdline')
                if not cmdline:
                    continue
                # Look for gunicorn worker processes running reader.main:app
                if (
                    any('gunicorn' in str(arg) for arg in cmdline)
                    and any('reader.main:app' in str(arg) for arg in cmdline)
                ):
                    # Check if process is listening or connected to the service port
                    connections = proc.info.get('connections')
                    if connections:
                        for conn in connections:
                            if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == self.service_port:
                                self.worker_procs.append(psutil.Process(proc.info['pid']))
                                break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not self.worker_procs:
            logger.warning(f"Could not find any Gunicorn worker processes listening on port {self.service_port}")
        else:
            logger.info(f"Found {len(self.worker_procs)} Gunicorn worker(s) for port {self.service_port}")
    
    def get_metrics(self) -> Optional[Tuple[float, List[WorkerMetrics]]]:
        """Get total memory usage across all Gunicorn workers, plus individual worker metrics."""
        self._find_worker_processes()  # Refresh in case of worker restarts
        if not self.worker_procs:
            return 0.0, []
        
        total_memory = 0.0
        worker_metrics = []
        
        for proc in self.worker_procs:
            try:
                mem = proc.memory_info().rss / 1024 / 1024  # MB
                total_memory += mem
                
                worker_metrics.append(WorkerMetrics(
                    pid=proc.pid,
                    memory_mb=mem
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return total_memory, worker_metrics

class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitor = ServiceMonitor()
        self.request_metrics: List[RequestMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.start_time = datetime.now()
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        
        # Test queries for variety
        self.test_queries = [
            "revenue",
            "cost center",
            "accounts receivable", 
            "net income",
            "cash flow",
            "operating expenses",
            "total assets",
            "liability",
            "equity",
            "profit margin"
        ]
    
    async def _make_request(
        self, 
        client: AsyncSearchIndexReaderClient, 
        query: str, 
        concurrent_count: int = 1
    ) -> RequestMetrics:
        """Make a single search request and record metrics."""
        start_time = time.time()
        retrieval_start = None
        retrieval_end = None
        status_code = None
        is_timeout = False
        is_throttled = False
        
        try:
            # Extract model_id from table_name (assumes format: hierarchies_{model_id})
            model_id = int(self.config.table_name.split('_')[1])
            
            # Time the retrieval phase
            retrieval_start = time.time()
            _ = await client.hybrid_search(
                tenant_id=self.config.tenant_id,
                model_id=model_id,
                query=query,
                limit=5
            )
            retrieval_end = time.time()
            
            total_time_ms = (time.time() - start_time) * 1000
            retrieval_time_ms = (retrieval_end - retrieval_start) * 1000 if retrieval_start and retrieval_end else None
            
            return RequestMetrics(
                timestamp=start_time,
                latency_ms=total_time_ms,
                success=True,
                concurrent_requests=concurrent_count,
                query=query,
                status_code=200,
                retrieval_time_ms=retrieval_time_ms,
                total_time_ms=total_time_ms,
                is_timeout=is_timeout,
                is_throttled=is_throttled
            )
            
        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            retrieval_time_ms = (retrieval_end - retrieval_start) * 1000 if retrieval_start and retrieval_end else None
            
            # Analyze error type
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                is_timeout = True
                status_code = 408
            elif '429' in error_str or 'too many requests' in error_str or 'throttle' in error_str:
                is_throttled = True
                status_code = 429
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            elif '4' in error_str[:10]:  # Check if error starts with 4xx
                status_code = 400
            elif '5' in error_str[:10]:  # Check if error starts with 5xx
                status_code = 500
            
            return RequestMetrics(
                timestamp=start_time,
                latency_ms=total_time_ms,
                success=False,
                error_message=str(e),
                concurrent_requests=concurrent_count,
                query=query,
                status_code=status_code,
                retrieval_time_ms=retrieval_time_ms,
                total_time_ms=total_time_ms,
                is_timeout=is_timeout,
                is_throttled=is_throttled
            )
    
    async def _system_monitor_task(self, stop_event: asyncio.Event):
        """Background task to monitor system metrics."""
        while not stop_event.is_set():
            metrics = self.monitor.get_metrics()
            if metrics:
                total_memory_mb, worker_metrics = metrics
                self.system_metrics.append(SystemMetrics(
                    timestamp=time.time(),
                    total_memory_mb=total_memory_mb,
                    concurrent_requests=0,  # Will be updated by main loop
                    worker_metrics=worker_metrics
                ))
            
            await asyncio.sleep(1)  # Monitor every second
    
    async def run_concurrent_test(self, concurrent_count: int, duration_seconds: int = 60) -> List[RequestMetrics]:
        """Run concurrent requests for specified duration."""
        console.print(f"Running concurrent test: {concurrent_count} concurrent requests for {duration_seconds}s")
        
        metrics = []
        auth = None
        if self.config.auth_user and self.config.auth_key:
            auth = (self.config.auth_user, self.config.auth_key)
        
        # Create client
        client = AsyncSearchIndexReaderClient(
            base_url=self.config.reader_url,
            timeout=self.config.request_timeout,
            auth=auth
        )
        
        try:
            end_time = time.time() + duration_seconds
            query_index = 0
            
            async with client:
                while time.time() < end_time:
                    # Create batch of concurrent requests
                    tasks = []
                    for _ in range(concurrent_count):
                        query = self.test_queries[query_index % len(self.test_queries)]
                        query_index += 1
                        
                        task = self._make_request(client, query, concurrent_count)
                        tasks.append(task)
                    
                    # Execute batch and collect results
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, RequestMetrics):
                            metrics.append(result)
                        else:
                            # Handle exceptions
                            metrics.append(RequestMetrics(
                                timestamp=time.time(),
                                latency_ms=0,
                                success=False,
                                error_message=str(result),
                                concurrent_requests=concurrent_count
                            ))
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in concurrent test: {e}")
        
        return metrics
    
    async def run_full_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark suite."""
        console.print("[bold blue]Starting Reader Service Performance Benchmark[/bold blue]")
        
        # Start system monitoring
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(self._system_monitor_task(stop_event))
        
        try:
            # Test different concurrency levels
            test_duration = 30  # seconds per test
            
            for concurrent_count in CONCURRENCY_LEVELS:
                # Update system metrics with current concurrency
                if self.system_metrics:
                    self.system_metrics[-1].concurrent_requests = concurrent_count
                
                # Run test
                batch_metrics = await self.run_concurrent_test(concurrent_count, test_duration)
                self.request_metrics.extend(batch_metrics)
                
                # Calculate current performance
                if batch_metrics:
                    successful = [m for m in batch_metrics if m.success]
                    if successful:
                        avg_latency = statistics.mean([m.latency_ms for m in successful])
                        success_rate = len(successful) / len(batch_metrics) * 100
                        
                        console.print(f"Concurrency {concurrent_count}: "
                                    f"Avg latency {avg_latency:.1f}ms, "
                                    f"Success rate {success_rate:.1f}%")
                        
                        # Detect breaking point (latency > 5000ms or success rate < 95%)
                        if avg_latency > 5000 or success_rate < 95:
                            console.print(f"[red]Breaking point detected at {concurrent_count} concurrent requests[/red]")
                            break
                
                # Brief pause between tests
                await asyncio.sleep(2)
            
        finally:
            # Stop monitoring
            stop_event.set()
            await monitor_task
        
        # Calculate final results
        return self._calculate_results()
    
    def _calculate_results(self) -> BenchmarkResults:
        """Calculate final benchmark results."""
        end_time = datetime.now()
        
        successful_requests = [m for m in self.request_metrics if m.success]
        failed_requests = [m for m in self.request_metrics if not m.success]
        total_requests = len(self.request_metrics)
        
        if successful_requests:
            latencies = [m.latency_ms for m in successful_requests]
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
            p999_latency = statistics.quantiles(latencies, n=1000)[998] if len(latencies) > 10 else latencies[0]
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = p999_latency = 0
        
        max_memory = max([m.total_memory_mb for m in self.system_metrics]) if self.system_metrics else 0
        
        # Calculate QPS (requests per second)
        duration_seconds = (end_time - self.start_time).total_seconds()
        total_qps = len(successful_requests) / duration_seconds if duration_seconds > 0 else 0
        
        # Calculate error rates
        error_rate = len(failed_requests) / total_requests * 100 if total_requests > 0 else 0
        timeout_requests = [m for m in self.request_metrics if m.is_timeout]
        timeout_rate = len(timeout_requests) / total_requests * 100 if total_requests > 0 else 0
        throttle_requests = [m for m in self.request_metrics if m.is_throttled]
        throttle_rate = len(throttle_requests) / total_requests * 100 if total_requests > 0 else 0
        
        # HTTP error rates
        http_4xx_requests = [m for m in self.request_metrics if m.status_code and 400 <= m.status_code < 500]
        http_4xx_rate = len(http_4xx_requests) / total_requests * 100 if total_requests > 0 else 0
        http_5xx_requests = [m for m in self.request_metrics if m.status_code and 500 <= m.status_code < 600]
        http_5xx_rate = len(http_5xx_requests) / total_requests * 100 if total_requests > 0 else 0
        
        # Detect breaking point
        breaking_point = None
        for concurrency in CONCURRENCY_LEVELS:
            concurrent_metrics = [m for m in successful_requests if m.concurrent_requests == concurrency]
            if concurrent_metrics:
                avg_lat = statistics.mean([m.latency_ms for m in concurrent_metrics])
                success_rate = len(concurrent_metrics) / len([m for m in self.request_metrics if m.concurrent_requests == concurrency])
                
                if avg_lat > 5000 or success_rate < 0.95:
                    breaking_point = concurrency
                    break
        
        return BenchmarkResults(
            config=self.config,
            request_metrics=self.request_metrics,
            system_metrics=self.system_metrics,
            start_time=self.start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            p999_latency_ms=p999_latency,
            max_memory_mb=max_memory,
            breaking_point_concurrent=breaking_point,
            total_qps=total_qps,
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            throttle_rate=throttle_rate,
            http_4xx_rate=http_4xx_rate,
            http_5xx_rate=http_5xx_rate
        )

class BenchmarkReporter:
    """Generate reports and visualizations from benchmark results."""
    
    def __init__(self, results: BenchmarkResults):
        self.results = results
    
    def print_summary(self):
        """Print summary results to console."""
        table = Table(title="Benchmark Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Request counts and rates
        table.add_row("Total Requests", str(self.results.total_requests))
        table.add_row("Successful Requests", str(self.results.successful_requests))
        table.add_row("Failed Requests", str(self.results.failed_requests))
        table.add_row("Success Rate", f"{(self.results.successful_requests/self.results.total_requests)*100:.1f}%")
        
        # Latency metrics (tail numbers matter more)
        table.add_row("Average Latency", f"{self.results.avg_latency_ms:.1f}ms")
        table.add_row("P50 Latency", f"{self.results.p50_latency_ms:.1f}ms")
        table.add_row("P95 Latency", f"{self.results.p95_latency_ms:.1f}ms")
        table.add_row("P99 Latency", f"{self.results.p99_latency_ms:.1f}ms")
        table.add_row("P999 Latency", f"{self.results.p999_latency_ms:.1f}ms")
        
        # Throughput and performance
        table.add_row("QPS (Requests/sec)", f"{self.results.total_qps:.1f}")
        table.add_row("Max Memory Usage", f"{self.results.max_memory_mb:.1f}MB")
        
        # Error rates and bottlenecks
        table.add_row("Error Rate", f"{self.results.error_rate:.2f}%")
        table.add_row("Timeout Rate", f"{self.results.timeout_rate:.2f}%")
        table.add_row("Throttle Rate (429)", f"{self.results.throttle_rate:.2f}%")
        table.add_row("HTTP 4xx Rate", f"{self.results.http_4xx_rate:.2f}%")
        table.add_row("HTTP 5xx Rate", f"{self.results.http_5xx_rate:.2f}%")
        
        if self.results.breaking_point_concurrent:
            table.add_row("Breaking Point", f"{self.results.breaking_point_concurrent} concurrent requests")
        else:
            table.add_row("Breaking Point", "Not reached")
        
        console.print(table)
    
    def save_csv_reports(self, output_dir: str):
        """Save detailed metrics to CSV files."""
        output_path = Path(output_dir)
        
        # Request metrics
        with open(output_path / "request_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "latency_ms", "success", "error_message", 
                "status_code", "concurrent_requests", "query", "is_timeout", 
                "is_throttled", "retrieval_time_ms", "total_time_ms"
            ])
            writer.writeheader()
            for metric in self.results.request_metrics:
                writer.writerow(asdict(metric))
        
        # System metrics  
        with open(output_path / "system_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "total_memory_mb", "concurrent_requests"
            ])
            writer.writeheader()
            for metric in self.results.system_metrics:
                writer.writerow({
                    "timestamp": metric.timestamp,
                    "total_memory_mb": metric.total_memory_mb,
                    "concurrent_requests": metric.concurrent_requests
                })
        
        # Individual worker metrics
        with open(output_path / "worker_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "pid", "memory_mb", "concurrent_requests"
            ])
            writer.writeheader()
            for metric in self.results.system_metrics:
                for worker in metric.worker_metrics:
                    writer.writerow({
                        "timestamp": metric.timestamp,
                        "pid": worker.pid,
                        "memory_mb": worker.memory_mb,
                        "concurrent_requests": metric.concurrent_requests
                    })
        
        # Summary
        with open(output_path / "summary.json", "w") as f:
            summary = {
                "total_requests": self.results.total_requests,
                "successful_requests": self.results.successful_requests,
                "failed_requests": self.results.failed_requests,
                "avg_latency_ms": self.results.avg_latency_ms,
                "p50_latency_ms": self.results.p50_latency_ms,
                "p95_latency_ms": self.results.p95_latency_ms,
                "p99_latency_ms": self.results.p99_latency_ms,
                "p999_latency_ms": self.results.p999_latency_ms,
                "total_qps": self.results.total_qps,
                "error_rate": self.results.error_rate,
                "timeout_rate": self.results.timeout_rate,
                "throttle_rate": self.results.throttle_rate,
                "http_4xx_rate": self.results.http_4xx_rate,
                "http_5xx_rate": self.results.http_5xx_rate,
                "max_memory_mb": self.results.max_memory_mb,
                "breaking_point_concurrent": self.results.breaking_point_concurrent,
                "start_time": self.results.start_time.isoformat(),
                "end_time": self.results.end_time.isoformat()
            }
            json.dump(summary, f, indent=2)
        
        console.print(f"[green]Reports saved to {output_dir}/[/green]")
    
    def generate_visualizations(self, output_dir: str):
        """Generate all performance visualizations."""
        output_path = Path(output_dir)
        
        # Convert metrics to DataFrames
        request_df = pd.DataFrame([asdict(m) for m in self.results.request_metrics])
        
        # Create system metrics dataframe (total metrics)
        system_df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'total_memory_mb': m.total_memory_mb,
            'concurrent_requests': m.concurrent_requests
        } for m in self.results.system_metrics])
        
        # Create worker metrics dataframe (individual worker metrics)
        worker_data = []
        for metric in self.results.system_metrics:
            for worker in metric.worker_metrics:
                worker_data.append({
                    'timestamp': metric.timestamp,
                    'pid': worker.pid,
                    'memory_mb': worker.memory_mb,
                    'concurrent_requests': metric.concurrent_requests
                })
        worker_df = pd.DataFrame(worker_data)
        
        # Convert timestamps
        if not request_df.empty:
            request_df['timestamp'] = pd.to_datetime(request_df['timestamp'], unit='s')
        if not system_df.empty:
            system_df['timestamp'] = pd.to_datetime(system_df['timestamp'], unit='s')
        if not worker_df.empty:
            worker_df['timestamp'] = pd.to_datetime(worker_df['timestamp'], unit='s')
        
        console.print("[yellow]Generating visualizations...[/yellow]")
        
        # Generate unified dashboard
        self._create_unified_dashboard(request_df, system_df, worker_df, output_path)
        
        console.print(f"[green]Visualizations saved to {output_dir}/[/green]")
    
    def _create_latency_vs_concurrency_plot(self, request_df: pd.DataFrame, output_path: Path):
        """Create latency vs concurrency plot with percentiles."""
        successful_df = request_df[request_df['success'] == True].copy()
        
        if successful_df.empty:
            return
        
        latency_stats = successful_df.groupby('concurrent_requests')['latency_ms'].agg([
            'mean', 'median',
            lambda x: x.quantile(0.95),
            lambda x: x.quantile(0.99),
            lambda x: x.quantile(0.999) if len(x) > 10 else x.max()
        ]).reset_index()
        
        latency_stats.columns = ['concurrent_requests', 'mean', 'median', 'p95', 'p99', 'p999']
        
        fig = go.Figure()
        
        # Add latency lines
        fig.add_trace(go.Scatter(
            x=latency_stats['concurrent_requests'],
            y=latency_stats['mean'],
            mode='lines+markers',
            name='Mean Latency',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=latency_stats['concurrent_requests'],
            y=latency_stats['p95'],
            mode='lines+markers',
            name='P95 Latency',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=latency_stats['concurrent_requests'],
            y=latency_stats['p99'],
            mode='lines+markers',
            name='P99 Latency',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Response Latency vs Concurrent Requests',
            xaxis_title='Concurrent Requests',
            yaxis_title='Latency (ms)',
            width=1200,
            height=700
        )
        
        fig.write_html(str(output_path / "latency_vs_concurrency.html"))
    
    def _create_memory_usage_plot(self, system_df: pd.DataFrame, worker_df: pd.DataFrame, output_path: Path):
        """Create memory and CPU usage plot showing individual workers."""
        if system_df.empty:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Memory Usage', 'Individual Worker Memory Usage', 
                          'Total CPU Usage', 'Individual Worker CPU Usage'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Total memory usage
        fig.add_trace(
            go.Scatter(
                x=system_df['timestamp'],
                y=system_df['total_memory_mb'],
                mode='lines',
                name='Total Memory (MB)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Individual worker memory usage
        if not worker_df.empty:
            for pid in worker_df['pid'].unique():
                worker_data = worker_df[worker_df['pid'] == pid]
                fig.add_trace(
                    go.Scatter(
                        x=worker_data['timestamp'],
                        y=worker_data['memory_mb'],
                        mode='lines',
                        name=f'Worker {pid}',
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Total CPU usage
        fig.add_trace(
            go.Scatter(
                x=system_df['timestamp'],
                y=system_df['total_cpu_percent'],
                mode='lines',
                name='Total CPU (%)',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Individual worker CPU usage
        if not worker_df.empty:
            for pid in worker_df['pid'].unique():
                worker_data = worker_df[worker_df['pid'] == pid]
                fig.add_trace(
                    go.Scatter(
                        x=worker_data['timestamp'],
                        y=worker_data['cpu_percent'],
                        mode='lines',
                        name=f'Worker {pid}',
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='System Resource Usage During Benchmark (Individual Workers)',
            height=800,
            width=1600,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_yaxes(title_text="CPU (%)", row=2, col=1)
        fig.update_yaxes(title_text="CPU (%)", row=2, col=2)
        
        fig.write_html(str(output_path / "memory_cpu_usage.html"))
    
    def _create_success_rate_plot(self, request_df: pd.DataFrame, output_path: Path):
        """Create success rate vs concurrency plot."""
        success_stats = request_df.groupby('concurrent_requests').agg({
            'success': ['count', 'sum']
        }).reset_index()
        
        success_stats.columns = ['concurrent_requests', 'total_requests', 'successful_requests']
        success_stats['success_rate'] = (success_stats['successful_requests'] / success_stats['total_requests']) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=success_stats['concurrent_requests'],
            y=success_stats['success_rate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='green', width=3)
        ))
        
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% Threshold")
        
        fig.update_layout(
            title='Request Success Rate vs Concurrent Requests',
            xaxis_title='Concurrent Requests',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[0, 105]),
            width=1000,
            height=600
        )
        
        fig.write_html(str(output_path / "success_rate.html"))
    
    def _create_qps_plot(self, request_df: pd.DataFrame, output_path: Path):
        """Create QPS vs concurrency plot."""
        successful_df = request_df[request_df['success'] == True].copy()
        
        if successful_df.empty:
            return
        
        qps_stats = successful_df.groupby('concurrent_requests').size().reset_index(name='total_requests')
        qps_stats['qps'] = qps_stats['total_requests'] / 30  # 30 second test duration
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=qps_stats['concurrent_requests'],
            y=qps_stats['qps'],
            mode='lines+markers',
            name='QPS (Requests/sec)',
            line=dict(color='purple', width=3)
        ))
        
        fig.update_layout(
            title='Throughput (QPS) vs Concurrent Requests',
            xaxis_title='Concurrent Requests',
            yaxis_title='Requests per Second (QPS)',
            width=1000,
            height=600
        )
        
        fig.write_html(str(output_path / "qps_vs_concurrency.html"))
    
    def _create_error_rates_plot(self, request_df: pd.DataFrame, output_path: Path):
        """Create error rates plot."""
        error_stats = request_df.groupby('concurrent_requests').agg({
            'success': 'count',
            'is_timeout': 'sum',
            'is_throttled': 'sum'
        }).reset_index()
        
        error_stats['timeout_rate'] = (error_stats['is_timeout'] / error_stats['success']) * 100
        error_stats['throttle_rate'] = (error_stats['is_throttled'] / error_stats['success']) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=error_stats['concurrent_requests'],
            y=error_stats['timeout_rate'],
            mode='lines+markers',
            name='Timeout Rate (%)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=error_stats['concurrent_requests'],
            y=error_stats['throttle_rate'],
            mode='lines+markers',
            name='Throttle Rate (429) (%)',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Error and Throttle Rates vs Concurrent Requests',
            xaxis_title='Concurrent Requests',
            yaxis_title='Error Rate (%)',
            width=1200,
            height=600
        )
        
        fig.write_html(str(output_path / "error_rates.html"))
    
    def _create_performance_dashboard(self, request_df: pd.DataFrame, system_df: pd.DataFrame, worker_df: pd.DataFrame, output_path: Path):
        """Create comprehensive performance dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latency Percentiles vs Concurrency',
                'QPS vs Concurrency',
                'Total Memory Usage Over Time',
                'Individual Worker Memory Usage'
            ),
            vertical_spacing=0.1
        )
        
        # 1. Latency Percentiles
        successful_df = request_df[request_df['success'] == True]
        if not successful_df.empty:
            latency_stats = successful_df.groupby('concurrent_requests')['latency_ms'].agg([
                lambda x: x.quantile(0.95),
                lambda x: x.quantile(0.99)
            ]).reset_index()
            latency_stats.columns = ['concurrent_requests', 'p95', 'p99']
            
            fig.add_trace(
                go.Scatter(x=latency_stats['concurrent_requests'], y=latency_stats['p95'],
                          mode='lines+markers', name='P95', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=latency_stats['concurrent_requests'], y=latency_stats['p99'],
                          mode='lines+markers', name='P99', line=dict(color='red')),
                row=1, col=1
            )
        
        # 2. QPS
        if not successful_df.empty:
            qps_stats = successful_df.groupby('concurrent_requests').size().reset_index(name='total_requests')
            qps_stats['qps'] = qps_stats['total_requests'] / 30
            fig.add_trace(
                go.Scatter(x=qps_stats['concurrent_requests'], y=qps_stats['qps'],
                          mode='lines+markers', name='QPS', line=dict(color='purple')),
                row=1, col=2
            )
        
        # 3. Total Memory Usage
        if not system_df.empty:
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['total_memory_mb'],
                          mode='lines', name='Total Memory (MB)', line=dict(color='red')),
                row=2, col=1
            )
        
        # 4. Individual Worker Memory Usage
        if not worker_df.empty:
            for pid in worker_df['pid'].unique():
                worker_data = worker_df[worker_df['pid'] == pid]
                fig.add_trace(
                    go.Scatter(x=worker_data['timestamp'], y=worker_data['memory_mb'],
                              mode='lines', name=f'Worker {pid}', line=dict(width=1)),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f'Performance Dashboard - {self.results.total_requests} Total Requests | QPS: {self.results.total_qps:.1f} | P99: {self.results.p99_latency_ms:.1f}ms',
            height=1000,
            width=1600
        )
        
        fig.write_html(str(output_path / "performance_dashboard.html"))
    
    def _create_unified_dashboard(self, request_df: pd.DataFrame, system_df: pd.DataFrame, worker_df: pd.DataFrame, output_path: Path):
        """Create comprehensive unified dashboard with all visualizations."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Latency Percentiles vs Concurrency',
                'QPS vs Concurrency',
                'Total Memory Usage Over Time',
                'Individual Worker Memory Usage',
                'Request Success Rate vs Concurrency',
                'Error and Throttle Rates vs Concurrency'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Latency Percentiles vs Concurrency
        successful_df = request_df[request_df['success'] == True]
        if not successful_df.empty:
            latency_stats = successful_df.groupby('concurrent_requests')['latency_ms'].agg([
                'mean', 'median',
                lambda x: x.quantile(0.95),
                lambda x: x.quantile(0.99),
                lambda x: x.quantile(0.999) if len(x) > 10 else x.max()
            ]).reset_index()
            latency_stats.columns = ['concurrent_requests', 'mean', 'median', 'p95', 'p99', 'p999']
            
            fig.add_trace(
                go.Scatter(x=latency_stats['concurrent_requests'], y=latency_stats['mean'],
                          mode='lines+markers', name='Mean', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=latency_stats['concurrent_requests'], y=latency_stats['p95'],
                          mode='lines+markers', name='P95', line=dict(color='orange', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=latency_stats['concurrent_requests'], y=latency_stats['p99'],
                          mode='lines+markers', name='P99', line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # 2. QPS vs Concurrency
        if not successful_df.empty:
            qps_stats = successful_df.groupby('concurrent_requests').size().reset_index(name='total_requests')
            qps_stats['qps'] = qps_stats['total_requests'] / 30  # 30 second test duration
            fig.add_trace(
                go.Scatter(x=qps_stats['concurrent_requests'], y=qps_stats['qps'],
                          mode='lines+markers', name='QPS', line=dict(color='purple', width=3)),
                row=1, col=2
            )
        
        # 3. Total Memory Usage Over Time
        if not system_df.empty:
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['total_memory_mb'],
                          mode='lines', name='Total Memory (MB)', line=dict(color='red', width=2)),
                row=2, col=1
            )
        
        # 4. Individual Worker Memory Usage
        if not worker_df.empty:
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
            for i, pid in enumerate(worker_df['pid'].unique()):
                worker_data = worker_df[worker_df['pid'] == pid]
                fig.add_trace(
                    go.Scatter(x=worker_data['timestamp'], y=worker_data['memory_mb'],
                              mode='lines', name=f'Worker {pid}', 
                              line=dict(color=colors[i % len(colors)], width=1)),
                    row=2, col=2
                )
        
        # 5. Request Success Rate vs Concurrency
        success_stats = request_df.groupby('concurrent_requests').agg({
            'success': ['count', 'sum']
        }).reset_index()
        success_stats.columns = ['concurrent_requests', 'total_requests', 'successful_requests']
        success_stats['success_rate'] = (success_stats['successful_requests'] / success_stats['total_requests']) * 100
        
        fig.add_trace(
            go.Scatter(x=success_stats['concurrent_requests'], y=success_stats['success_rate'],
                      mode='lines+markers', name='Success Rate', line=dict(color='green', width=3)),
            row=3, col=1
        )
        
        # Add 95% threshold line for success rate
        fig.add_hline(y=95, line_dash="dash", line_color="red", row=3, col=1)
        
        # 6. Error and Throttle Rates vs Concurrency
        error_stats = request_df.groupby('concurrent_requests').agg({
            'success': 'count',
            'is_timeout': 'sum',
            'is_throttled': 'sum'
        }).reset_index()
        error_stats['timeout_rate'] = (error_stats['is_timeout'] / error_stats['success']) * 100
        error_stats['throttle_rate'] = (error_stats['is_throttled'] / error_stats['success']) * 100
        
        fig.add_trace(
            go.Scatter(x=error_stats['concurrent_requests'], y=error_stats['timeout_rate'],
                      mode='lines+markers', name='Timeout Rate (%)', line=dict(color='red', width=2)),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=error_stats['concurrent_requests'], y=error_stats['throttle_rate'],
                      mode='lines+markers', name='Throttle Rate (%)', line=dict(color='orange', width=2)),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Performance Benchmark Dashboard - {self.results.total_requests} Total Requests | QPS: {self.results.total_qps:.1f} | P99: {self.results.p99_latency_ms:.1f}ms | Max Memory: {self.results.max_memory_mb:.1f}MB',
            height=1200,
            width=1800,
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Concurrent Requests", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        
        fig.update_xaxes(title_text="Concurrent Requests", row=1, col=2)
        fig.update_yaxes(title_text="QPS", row=1, col=2)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
        
        fig.update_xaxes(title_text="Concurrent Requests", row=3, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Concurrent Requests", row=3, col=2)
        fig.update_yaxes(title_text="Error Rate (%)", row=3, col=2)
        
        # Save unified dashboard
        fig.write_html(str(output_path / "benchmark_dashboard.html"))
        console.print(f"[green]Unified dashboard saved to {output_path}/benchmark_dashboard.html[/green]")

@click.command()
@click.option("--tenant-id", default="1466591000091951104", help="Tenant ID to test")
@click.option("--table-name", default="hierarchies_1466593582621392896", help="Table name to test")
@click.option("--reader-url", default="http://localhost:8002", help="Reader service URL")
@click.option("--auth-user", default=None, help="Auth username (or set VENA_USER env var)")
@click.option("--auth-key", default=None, help="Auth key (or set VENA_KEY env var)")
@click.option("--output-dir", default="benchmark_results", help="Output directory for results")
@click.option("--max-concurrent", default=100, help="Maximum concurrent requests to test")
@click.option("--timeout", default=30.0, help="Request timeout in seconds")
def main(tenant_id, table_name, reader_url, auth_user, auth_key, output_dir, max_concurrent, timeout):
    """Run performance benchmark against the reader service."""
    
    # Use environment variables if auth not provided
    if not auth_user:
        auth_user = os.getenv("VENA_USER")
    if not auth_key:
        auth_key = os.getenv("VENA_KEY")
    
    config = BenchmarkConfig(
        tenant_id=tenant_id,
        table_name=table_name,
        reader_url=reader_url,
        auth_user=auth_user,
        auth_key=auth_key,
        output_dir=output_dir,
        max_concurrent=max_concurrent,
        request_timeout=timeout
    )
    
    console.print(f"[bold green]Reader Service Benchmark Configuration[/bold green]")
    console.print(f"Tenant ID: {config.tenant_id}")
    console.print(f"Table: {config.table_name}")
    console.print(f"Service URL: {config.reader_url}")
    console.print(f"Output Directory: {config.output_dir}")
    console.print()
    
    try:
        # Run benchmark
        runner = BenchmarkRunner(config)
        results = asyncio.run(runner.run_full_benchmark())
        
        # Generate reports
        reporter = BenchmarkReporter(results)
        reporter.print_summary()
        reporter.save_csv_reports(config.output_dir)
        reporter.generate_visualizations(config.output_dir)
        
        console.print(f"[bold green]Benchmark completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise

if __name__ == "__main__":
    main()