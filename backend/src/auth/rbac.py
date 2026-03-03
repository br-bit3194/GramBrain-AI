"""
Role-Based Access Control (RBAC)
"""
from enum import Enum
from typing import Set, Dict

class Role(str, Enum):
    FARMER = "farmer"
    VILLAGE_LEADER = "village_leader"
    POLICYMAKER = "policymaker"
    CONSUMER = "consumer"
    ADMIN = "admin"

class Permission(str, Enum):
    # Farm permissions
    VIEW_OWN_FARM = "view_own_farm"
    EDIT_OWN_FARM = "edit_own_farm"
    VIEW_ALL_FARMS = "view_all_farms"
    
    # Query permissions
    MAKE_QUERY = "make_query"
    VIEW_OWN_QUERIES = "view_own_queries"
    VIEW_ALL_QUERIES = "view_all_queries"
    
    # Product permissions
    CREATE_PRODUCT = "create_product"
    VIEW_OWN_PRODUCTS = "view_own_products"
    VIEW_ALL_PRODUCTS = "view_all_products"
    BUY_PRODUCTS = "buy_products"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SYSTEM = "manage_system"

# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.FARMER: {
        Permission.VIEW_OWN_FARM,
        Permission.EDIT_OWN_FARM,
        Permission.MAKE_QUERY,
        Permission.VIEW_OWN_QUERIES,
        Permission.CREATE_PRODUCT,
        Permission.VIEW_OWN_PRODUCTS,
        Permission.VIEW_ALL_PRODUCTS,
    },
    Role.VILLAGE_LEADER: {
        Permission.VIEW_OWN_FARM,
        Permission.EDIT_OWN_FARM,
        Permission.VIEW_ALL_FARMS,
        Permission.MAKE_QUERY,
        Permission.VIEW_OWN_QUERIES,
        Permission.VIEW_ALL_QUERIES,
        Permission.VIEW_ALL_PRODUCTS,
        Permission.VIEW_ANALYTICS,
    },
    Role.POLICYMAKER: {
        Permission.VIEW_ALL_FARMS,
        Permission.VIEW_ALL_QUERIES,
        Permission.VIEW_ALL_PRODUCTS,
        Permission.VIEW_ANALYTICS,
    },
    Role.CONSUMER: {
        Permission.VIEW_ALL_PRODUCTS,
        Permission.BUY_PRODUCTS,
    },
    Role.ADMIN: set(Permission),  # All permissions
}

def has_permission(role: str, permission: Permission) -> bool:
    """Check if role has permission"""
    try:
        role_enum = Role(role)
        return permission in ROLE_PERMISSIONS.get(role_enum, set())
    except ValueError:
        return False

def get_role_permissions(role: str) -> Set[Permission]:
    """Get all permissions for a role"""
    try:
        role_enum = Role(role)
        return ROLE_PERMISSIONS.get(role_enum, set())
    except ValueError:
        return set()
