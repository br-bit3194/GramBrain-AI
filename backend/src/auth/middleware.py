"""
FastAPI authentication and authorization middleware
"""
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from .auth_service import AuthService
from .rbac import has_permission, Permission

security = HTTPBearer()
auth_service = AuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role")
        if not has_permission(role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required permission: {permission.value}"
            )
        return current_user
    return permission_checker

def require_role(*allowed_roles: str):
    """Decorator to require specific role(s)"""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
            )
        return current_user
    return role_checker
