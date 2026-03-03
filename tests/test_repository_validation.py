"""Tests for repository data validation using Pydantic schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

from data.models import User, UserRole
from data.schemas import UserSchema, UserCreateSchema


class TestUserSchemaValidation:
    """Test User schema validation."""
    
    def test_valid_user_schema(self):
        """Test that valid user data passes validation."""
        user_data = {
            'user_id': 'user_123',
            'phone_number': '1234567890',
            'name': 'Test User',
            'language_preference': 'en',
            'role': UserRole.FARMER,
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'metadata': {'key': 'value'}
        }
        
        # Should not raise
        schema = UserSchema(**user_data)
        assert schema.user_id == 'user_123'
        assert schema.phone_number == '1234567890'
    
    def test_invalid_phone_number(self):
        """Test that invalid phone number fails validation."""
        user_data = {
            'user_id': 'user_123',
            'phone_number': 'abc',  # Invalid: not digits
            'name': 'Test User',
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserSchema(**user_data)
        
        assert 'phone_number' in str(exc_info.value)
    
    def test_empty_name(self):
        """Test that empty name fails validation."""
        user_data = {
            'user_id': 'user_123',
            'phone_number': '1234567890',
            'name': '   ',  # Empty after strip
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserSchema(**user_data)
        
        assert 'name' in str(exc_info.value)
    
    def test_short_phone_number(self):
        """Test that short phone number fails validation."""
        user_data = {
            'user_id': 'user_123',
            'phone_number': '123',  # Too short
            'name': 'Test User',
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserSchema(**user_data)
        
        assert 'phone_number' in str(exc_info.value)
    
    def test_user_create_schema(self):
        """Test UserCreateSchema for API requests."""
        create_data = {
            'phone_number': '1234567890',
            'name': 'New User',
            'language_preference': 'hi',
            'role': UserRole.FARMER
        }
        
        # Should not raise
        schema = UserCreateSchema(**create_data)
        assert schema.phone_number == '1234567890'
        assert schema.language_preference == 'hi'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
