"""
Gunicorn configuration for search-index-reader service.

This configuration optimizes the reader service for high-frequency search operations
while avoiding fork-related issues with LanceDB by using async workers.
"""

import os

# Server socket - Optimized for high concurrency
bind = "0.0.0.0:8002"
backlog = 4096  # Higher backlog for more concurrent connections

# Worker processes - Optimized for 15 QPS + <500ms latency target
# Balanced for Azure workloads and low latency requirements
workers = 8  # Maintained for sufficient concurrency
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000  # Async concurrency per worker
max_requests = 1000  # Increased for better connection reuse
max_requests_jitter = 100

# Timeouts - Optimized for <500ms latency target
timeout = 60    # Reduced for faster timeout detection
keepalive = 2   # Shorter keepalive for faster connection turnover
graceful_timeout = 15  # Faster graceful shutdown

# Logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "search-index-reader"

# Performance optimizations
# Enable preload_app for faster worker startup and reduced memory usage
preload_app = False  # Safe for read-only LanceDB operations
enable_stdio_inheritance = True

# Memory management - Optimized for Azure and low latency
max_worker_memory = 1536 * 1024 * 1024  # 1.5GB per worker for Azure workloads and caching

# Security
forwarded_allow_ips = "*"
secure_scheme_headers = {
    "X-FORWARDED-PROTOCOL": "ssl",
    "X-FORWARDED-PROTO": "https",
    "X-FORWARDED-SSL": "on"
}

def post_fork(server, worker):
    """Called after a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    """Called before a worker is forked."""
    pass

def when_ready(server):
    """Called when the server is ready to serve requests."""
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    """Called when a worker receives the INT or QUIT signal."""
    worker.log.info("Worker received INT or QUIT signal")

def on_exit(server):
    """Called when the server exits."""
    server.log.info("Server is shutting down")