"""Auth header extraction middleware for FastAPI applications."""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Optional


class AuthHeaderMiddleware:
    """Middleware to extract auth header from requests and pass to Vena API."""
    
    def __init__(self):
        pass
    
    async def __call__(self, request: Request, call_next):
        """Extract auth header from request and store for downstream services."""
        # Only apply auth header extraction to API endpoints
        if not request.url.path.startswith("/api"):
            return await call_next(request)
        
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing Authorization header"},
                headers={"WWW-Authenticate": "Basic"},
            )
        
        # Store auth header in request state for passing to vena_service
        request.state.auth_header = authorization
        
        return await call_next(request)


def get_auth_header(request: Request) -> Optional[str]:
    """Extract the auth header from the request state."""
    return getattr(request.state, "auth_header", None)