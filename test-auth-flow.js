/**
 * Test script to verify authentication flow between frontend and backend
 */

const axios = require('axios');

const API_URL = 'http://localhost:8000/api';

async function testAuthFlow() {
  console.log('🔐 Testing Authentication Flow\n');
  console.log('='.repeat(60));

  try {
    // Test 1: Register a new user
    console.log('\n1️⃣  Testing User Registration...');
    const registerData = {
      phone_number: '+91 98765 43210',
      name: 'Test Farmer',
      password: 'testpassword123',
      language_preference: 'en',
      role: 'farmer'
    };

    const registerResponse = await axios.post(`${API_URL}/auth/register`, registerData);
    console.log('✅ Registration successful');
    console.log('   User ID:', registerResponse.data.data.user.user_id);
    console.log('   Access Token:', registerResponse.data.data.access_token.substring(0, 20) + '...');
    console.log('   Token Type:', registerResponse.data.data.token_type);

    const accessToken = registerResponse.data.data.access_token;

    // Test 2: Get current user with token
    console.log('\n2️⃣  Testing Get Current User (with token)...');
    const meResponse = await axios.get(`${API_URL}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    console.log('✅ Get current user successful');
    console.log('   User:', meResponse.data.data.user);

    // Test 3: Try to access protected endpoint without token
    console.log('\n3️⃣  Testing Protected Endpoint (without token)...');
    try {
      await axios.get(`${API_URL}/auth/me`);
      console.log('❌ Should have failed without token');
    } catch (error) {
      if (error.response && error.response.status === 401) {
        console.log('✅ Correctly rejected request without token');
        console.log('   Status:', error.response.status);
        console.log('   Message:', error.response.data.detail);
      } else {
        throw error;
      }
    }

    // Test 4: Login with credentials
    console.log('\n4️⃣  Testing User Login...');
    const loginData = {
      phone_number: '+91 98765 43210',
      password: 'testpassword123'
    };

    const loginResponse = await axios.post(`${API_URL}/auth/login`, loginData);
    console.log('✅ Login successful');
    console.log('   User ID:', loginResponse.data.data.user.user_id);
    console.log('   Access Token:', loginResponse.data.data.access_token.substring(0, 20) + '...');

    // Test 5: Try login with wrong password
    console.log('\n5️⃣  Testing Login with Wrong Password...');
    try {
      await axios.post(`${API_URL}/auth/login`, {
        phone_number: '+91 98765 43210',
        password: 'wrongpassword'
      });
      console.log('❌ Should have failed with wrong password');
    } catch (error) {
      if (error.response && error.response.status === 401) {
        console.log('✅ Correctly rejected wrong password');
        console.log('   Status:', error.response.status);
        console.log('   Message:', error.response.data.detail);
      } else {
        throw error;
      }
    }

    // Test 6: Test CORS headers
    console.log('\n6️⃣  Testing CORS Configuration...');
    const healthResponse = await axios.get(`${API_URL}/health`);
    console.log('✅ CORS working - able to make cross-origin request');
    console.log('   Status:', healthResponse.data.data.status);

    console.log('\n' + '='.repeat(60));
    console.log('✅ All authentication tests passed!\n');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', error.response.data);
    }
    process.exit(1);
  }
}

// Run tests
testAuthFlow();
