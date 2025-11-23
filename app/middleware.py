# app/middleware.py

from fastapi import Request

from app.metrics import ERROR_COUNT, IN_FLIGHT, REQUEST_COUNT


async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    # Track in-flight requests
    IN_FLIGHT.labels(endpoint=endpoint).inc()

    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception as exc:
        # Count exceptions
        ERROR_COUNT.labels(endpoint=endpoint, exception_type=type(exc).__name__).inc()
        raise
    finally:
        # Decrease in-flight
        IN_FLIGHT.labels(endpoint=endpoint).dec()

        # Count request (always increment)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()

    return response
