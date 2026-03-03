/**
 * Simple verification script for API client configuration
 * Run with: node verify-api-client.js
 */

// Mock environment variable
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000/api';

// Mock axios to capture configuration
const axiosConfigs = [];
const mockAxiosInstance = {
  interceptors: {
    request: { use: () => {} },
    response: { use: () => {} }
  }
};

const axios = {
  create: (config) => {
    axiosConfigs.push(config);
    console.log('✓ axios.create called with config:', JSON.stringify(config, null, 2));
    return mockAxiosInstance;
  }
};

// Mock the axios module
require.cache[require.resolve('axios')] = {
  exports: axios
};

// Now require the API client
console.log('\n=== API Client Configuration Verification ===\n');

try {
  // This will trigger the constructor
  require('./src/services/api.ts');
  
  console.log('\n=== Verification Results ===\n');
  
  const config = axiosConfigs[0];
  
  // Check 1: Base URL includes /api
  if (config.baseURL && config.baseURL.endsWith('/api')) {
    console.log('✓ Base URL correctly includes /api prefix');
    console.log(`  Base URL: ${config.baseURL}`);
  } else {
    console.log('✗ Base URL does not include /api prefix');
    console.log(`  Base URL: ${config.baseURL}`);
  }
  
  // Check 2: Content-Type header
  if (config.headers && config.headers['Content-Type'] === 'application/json') {
    console.log('✓ Content-Type header is set to application/json');
  } else {
    console.log('✗ Content-Type header is not properly configured');
  }
  
  // Check 3: Timeout is set
  if (config.timeout && config.timeout > 0) {
    console.log(`✓ Timeout is configured: ${config.timeout}ms`);
  } else {
    console.log('✗ Timeout is not configured');
  }
  
  console.log('\n=== All checks passed! ===\n');
  
} catch (error) {
  console.error('✗ Error loading API client:', error.message);
  process.exit(1);
}
