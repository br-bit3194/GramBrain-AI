// Test registration and login flow
const axios = require('axios');

const API_URL = 'http://localhost:8000/api';

async function testRegistration() {
  console.log('Testing Registration Flow...\n');
  
  const testUser = {
    phone_number: '+91 9876543210',
    name: 'Test Farmer',
    password: 'test123',
    language_preference: 'en',
    role: 'farmer'
  };

  try {
    // Test registration
    console.log('1. Registering new user...');
    const registerResponse = await axios.post(`${API_URL}/auth/register`, testUser);
    console.log('✓ Registration successful');
    console.log('User:', registerResponse.data.data.user);
    console.log('Access Token:', registerResponse.data.data.access_token.substring(0, 20) + '...');
    
    const accessToken = registerResponse.data.data.access_token;
    
    // Test login
    console.log('\n2. Testing login...');
    const loginResponse = await axios.post(`${API_URL}/auth/login`, {
      phone_number: testUser.phone_number,
      password: testUser.password
    });
    console.log('✓ Login successful');
    console.log('User:', loginResponse.data.data.user);
    
    // Test authenticated endpoint
    console.log('\n3. Testing authenticated endpoint...');
    const meResponse = await axios.get(`${API_URL}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    console.log('✓ Authenticated request successful');
    console.log('User:', meResponse.data.data.user);
    
    console.log('\n✓ All tests passed!');
    
  } catch (error) {
    console.error('✗ Test failed:');
    if (error.response) {
      console.error('Status:', error.response.status);
      console.error('Data:', error.response.data);
    } else {
      console.error('Error:', error.message);
    }
  }
}

testRegistration();
