/**
 * Frontend-Backend Connection Test Script
 * Tests all major API endpoints and connection health
 */

const axios = require('axios');

const BACKEND_URL = 'http://localhost:8000';
const API_URL = `${BACKEND_URL}/api`;

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSuccess(message) {
  log(`✓ ${message}`, 'green');
}

function logError(message) {
  log(`✗ ${message}`, 'red');
}

function logInfo(message) {
  log(`ℹ ${message}`, 'cyan');
}

function logWarning(message) {
  log(`⚠ ${message}`, 'yellow');
}

// Test results tracker
const results = {
  passed: 0,
  failed: 0,
  tests: [],
};

async function runTest(name, testFn) {
  try {
    log(`\n${name}...`, 'blue');
    await testFn();
    logSuccess(`${name} - PASSED`);
    results.passed++;
    results.tests.push({ name, status: 'PASSED' });
  } catch (error) {
    logError(`${name} - FAILED`);
    logError(`  Error: ${error.message}`);
    results.failed++;
    results.tests.push({ name, status: 'FAILED', error: error.message });
  }
}

// Test 1: Backend Server Running
async function testBackendRunning() {
  const response = await axios.get(`${BACKEND_URL}/health`, { timeout: 5000 });
  if (response.status !== 200) {
    throw new Error(`Expected status 200, got ${response.status}`);
  }
  logInfo(`  Backend is running on ${BACKEND_URL}`);
  logInfo(`  Status: ${response.data.data.status}`);
  logInfo(`  Agents: ${response.data.data.agents.length} registered`);
}

// Test 2: CORS Configuration
async function testCORS() {
  const response = await axios.get(`${BACKEND_URL}/health`, {
    headers: {
      'Origin': 'http://localhost:3000',
    },
  });
  
  if (response.status !== 200) {
    throw new Error('CORS request failed');
  }
  logInfo('  CORS is properly configured for localhost:3000');
}

// Test 3: User Registration
async function testUserRegistration() {
  const testUser = {
    phone_number: `+91${Math.floor(Math.random() * 10000000000)}`,
    name: 'Test User',
    password: 'testpass123',
    language_preference: 'en',
    role: 'farmer',
  };

  const response = await axios.post(`${API_URL}/auth/register`, testUser);
  
  if (response.data.status !== 'success') {
    throw new Error('Registration failed');
  }
  
  if (!response.data.data.access_token) {
    throw new Error('No access token returned');
  }
  
  logInfo(`  User registered: ${testUser.phone_number}`);
  logInfo(`  Access token received: ${response.data.data.access_token.substring(0, 20)}...`);
  
  return {
    user: response.data.data.user,
    token: response.data.data.access_token,
  };
}

// Test 4: User Login
async function testUserLogin() {
  // First register a user
  const testUser = {
    phone_number: `+91${Math.floor(Math.random() * 10000000000)}`,
    name: 'Login Test User',
    password: 'testpass123',
  };

  await axios.post(`${API_URL}/auth/register`, testUser);

  // Now login
  const loginResponse = await axios.post(`${API_URL}/auth/login`, {
    phone_number: testUser.phone_number,
    password: testUser.password,
  });

  if (loginResponse.data.status !== 'success') {
    throw new Error('Login failed');
  }

  if (!loginResponse.data.data.access_token) {
    throw new Error('No access token returned');
  }

  logInfo(`  Login successful for: ${testUser.phone_number}`);
  logInfo(`  Token type: ${loginResponse.data.data.token_type}`);
  
  return loginResponse.data.data;
}

// Test 5: Protected Route with Token
async function testProtectedRoute() {
  // Register and get token
  const { token } = await testUserRegistration();

  // Access protected route
  const response = await axios.get(`${API_URL}/auth/me`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('Protected route access failed');
  }

  logInfo('  Successfully accessed protected route with token');
  logInfo(`  User ID: ${response.data.data.user.user_id}`);
}

// Test 6: Create Farm
async function testCreateFarm() {
  const { user, token } = await testUserRegistration();

  const farmData = {
    owner_id: user.user_id,
    latitude: 28.7041,
    longitude: 77.1025,
    area_hectares: 5.5,
    soil_type: 'loamy',
    irrigation_type: 'drip',
  };

  const response = await axios.post(`${API_URL}/farms`, farmData, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('Farm creation failed');
  }

  logInfo(`  Farm created: ${response.data.data.farm.farm_id}`);
  logInfo(`  Location: ${farmData.latitude}, ${farmData.longitude}`);
  logInfo(`  Area: ${farmData.area_hectares} hectares`);
  
  return response.data.data.farm;
}

// Test 7: Process AI Query
async function testProcessQuery() {
  const { user, token } = await testUserRegistration();
  const farm = await testCreateFarm();

  const queryData = {
    user_id: user.user_id,
    query_text: 'Should I irrigate my wheat field today?',
    farm_id: farm.farm_id,
    latitude: 28.7041,
    longitude: 77.1025,
    farm_size_hectares: 5.5,
    crop_type: 'wheat',
    growth_stage: 'tillering',
    soil_type: 'loamy',
    language: 'en',
  };

  const response = await axios.post(`${API_URL}/query`, queryData, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('Query processing failed');
  }

  logInfo(`  Query processed successfully`);
  logInfo(`  Recommendation ID: ${response.data.data.recommendation.recommendation_id}`);
  logInfo(`  Confidence: ${(response.data.data.recommendation.confidence * 100).toFixed(1)}%`);
  logInfo(`  Recommendation: ${response.data.data.recommendation.recommendation_text.substring(0, 100)}...`);
}

// Test 8: Search Knowledge
async function testSearchKnowledge() {
  const { token } = await testUserRegistration();

  const response = await axios.get(`${API_URL}/knowledge/search`, {
    params: {
      query: 'wheat irrigation',
      top_k: 5,
    },
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('Knowledge search failed');
  }

  logInfo(`  Knowledge search completed`);
  logInfo(`  Results found: ${response.data.data.count}`);
}

// Test 9: Error Handling (401 Unauthorized)
async function testErrorHandling401() {
  try {
    await axios.get(`${API_URL}/auth/me`, {
      headers: {
        'Authorization': 'Bearer invalid_token',
      },
    });
    throw new Error('Should have thrown 401 error');
  } catch (error) {
    if (error.response && error.response.status === 401) {
      logInfo('  401 error properly returned for invalid token');
    } else {
      throw error;
    }
  }
}

// Test 10: Error Handling (422 Validation)
async function testErrorHandling422() {
  try {
    await axios.post(`${API_URL}/auth/register`, {
      // Missing required fields
      name: 'Test',
    });
    throw new Error('Should have thrown 422 error');
  } catch (error) {
    if (error.response && error.response.status === 422) {
      logInfo('  422 validation error properly returned');
      logInfo(`  Errors: ${JSON.stringify(error.response.data.errors || error.response.data.detail)}`);
    } else {
      throw error;
    }
  }
}

// Test 11: List User Farms
async function testListUserFarms() {
  const { user, token } = await testUserRegistration();
  await testCreateFarm();

  const response = await axios.get(`${API_URL}/users/${user.user_id}/farms`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('List farms failed');
  }

  logInfo(`  Farms listed: ${response.data.data.farms.length}`);
}

// Test 12: Create Product
async function testCreateProduct() {
  const { user, token } = await testUserRegistration();
  const farm = await testCreateFarm();

  const productData = {
    farmer_id: user.user_id,
    farm_id: farm.farm_id,
    product_type: 'vegetables',
    name: 'Organic Tomatoes',
    quantity_kg: 100,
    price_per_kg: 50,
    harvest_date: new Date().toISOString(),
  };

  const response = await axios.post(`${API_URL}/products`, productData, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (response.data.status !== 'success') {
    throw new Error('Product creation failed');
  }

  logInfo(`  Product created: ${response.data.data.product.name}`);
  logInfo(`  Product ID: ${response.data.data.product.product_id}`);
}

// Main test runner
async function runAllTests() {
  log('\n╔════════════════════════════════════════════════════════════╗', 'cyan');
  log('║     GramBrain AI - Frontend-Backend Connection Test       ║', 'cyan');
  log('╚════════════════════════════════════════════════════════════╝', 'cyan');

  log('\nStarting connection tests...\n', 'yellow');

  await runTest('Test 1: Backend Server Running', testBackendRunning);
  await runTest('Test 2: CORS Configuration', testCORS);
  await runTest('Test 3: User Registration', testUserRegistration);
  await runTest('Test 4: User Login', testUserLogin);
  await runTest('Test 5: Protected Route with Token', testProtectedRoute);
  await runTest('Test 6: Create Farm', testCreateFarm);
  await runTest('Test 7: Process AI Query', testProcessQuery);
  await runTest('Test 8: Search Knowledge', testSearchKnowledge);
  await runTest('Test 9: Error Handling (401)', testErrorHandling401);
  await runTest('Test 10: Error Handling (422)', testErrorHandling422);
  await runTest('Test 11: List User Farms', testListUserFarms);
  await runTest('Test 12: Create Product', testCreateProduct);

  // Print summary
  log('\n╔════════════════════════════════════════════════════════════╗', 'cyan');
  log('║                      TEST SUMMARY                          ║', 'cyan');
  log('╚════════════════════════════════════════════════════════════╝', 'cyan');
  
  log(`\nTotal Tests: ${results.passed + results.failed}`, 'blue');
  logSuccess(`Passed: ${results.passed}`);
  
  if (results.failed > 0) {
    logError(`Failed: ${results.failed}`);
    log('\nFailed Tests:', 'red');
    results.tests
      .filter(t => t.status === 'FAILED')
      .forEach(t => {
        logError(`  - ${t.name}`);
        logError(`    ${t.error}`);
      });
  } else {
    log('\n🎉 All tests passed! Frontend-Backend connection is working perfectly!', 'green');
  }

  log('\n');
  
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runAllTests().catch(error => {
  logError(`\nFatal error: ${error.message}`);
  logError(error.stack);
  process.exit(1);
});
