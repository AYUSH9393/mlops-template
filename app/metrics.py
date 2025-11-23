# app/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# -----------------------------------------
# Count total API requests (method + path + status)
# -----------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

# -----------------------------------------
# Count error responses by endpoint + exception type
# -----------------------------------------
ERROR_COUNT = Counter(
    "api_request_errors_total",
    "Total number of error responses",
    ["endpoint", "exception_type"],
)

# -----------------------------------------
# Track number of in-flight requests (currently processing)
# -----------------------------------------
IN_FLIGHT = Gauge(
    "in_flight_requests",
    "Number of in-flight API requests being processed",
    ["endpoint"],
)

# -----------------------------------------
# Track model inference latency 
# (histogram allows P95/P99 latency calculation)
# -----------------------------------------
INFERENCE_LATENCY = Histogram(
    "inference_seconds",
    "Inference latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
