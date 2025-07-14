"""
Gunicorn configuration for search-index-writer service.

This configuration optimizes the writer service for data ingestion and table creation operations.
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8001"
backlog = 2048

# Worker processes
# Writer service typically handles fewer concurrent requests but longer-running operations
# Using fewer workers but with higher memory and timeout limits
workers = max(multiprocessing.cpu_count(), 2)
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 100  # Lower due to long-running operations
max_requests_jitter = 10

# Timeouts
# Very high timeout for data ingestion and table creation operations
timeout = 600  # 10 minutes for large data processing
keepalive = 2
graceful_timeout = 60

# Logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "search-index-writer"

# Performance
preload_app = True
enable_stdio_inheritance = True

# Memory management
max_worker_memory = 1024 * 1024 * 1024  # 1GB per worker for data processing

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