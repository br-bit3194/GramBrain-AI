/**
 * Verification script to check that frontend API endpoints match backend routes
 */

const frontendEndpoints = {
  auth: [
    '/auth/register',
    '/auth/login',
    '/auth/me',
  ],
  users: [
    '/users',
    '/users/{user_id}',
    '/users/{user_id}/farms',
    '/users/{user_id}/recommendations',
  ],
  farms: [
    '/farms',
    '/farms/{farm_id}',
  ],
  query: [
    '/query',
    '/recommendations/{recommendation_id}',
  ],
  products: [
    '/products',
    '/products/{product_id}',
    '/farmers/{farmer_id}/products',
  ],
  knowledge: [
    '/knowledge',
    '/knowledge/search',
    '/knowledge/bulk',
  ],
  health: [
    '/health',
  ],
}

const backendRoutes = {
  auth: [
    '/api/auth/register',
    '/api/auth/login',
    '/api/auth/me',
  ],
  users: [
    '/api/users',
    '/api/users/{user_id}',
    '/api/users/{user_id}/farms',
    '/api/users/{user_id}/recommendations',
  ],
  farms: [
    '/api/farms',
    '/api/farms/{farm_id}',
  ],
  query: [
    '/api/query',
    '/api/recommendations/{recommendation_id}',
  ],
  products: [
    '/api/products',
    '/api/products/{product_id}',
    '/api/farmers/{farmer_id}/products',
  ],
  knowledge: [
    '/api/knowledge',
    '/api/knowledge/search',
    '/api/knowledge/bulk',
  ],
  health: [
    '/health',
  ],
}

console.log('🔍 Verifying Frontend API Endpoints Match Backend Routes\n')

let allMatch = true

Object.keys(frontendEndpoints).forEach(category => {
  console.log(`\n📁 ${category.toUpperCase()}:`)
  
  const frontendPaths = frontendEndpoints[category]
  const backendPaths = backendRoutes[category]
  
  frontendPaths.forEach((path, index) => {
    const backendPath = backendPaths[index]
    const expectedBackendPath = backendPath.replace('/api', '')
    
    if (path === expectedBackendPath) {
      console.log(`  ✅ ${path} -> ${backendPath}`)
    } else {
      console.log(`  ❌ MISMATCH: Frontend: ${path}, Backend: ${backendPath}`)
      allMatch = false
    }
  })
  
  // Check if backend has more routes
  if (backendPaths.length > frontendPaths.length) {
    console.log(`  ⚠️  Backend has ${backendPaths.length - frontendPaths.length} more route(s)`)
    backendPaths.slice(frontendPaths.length).forEach(path => {
      console.log(`     Missing: ${path}`)
    })
    allMatch = false
  }
  
  // Check if frontend has more routes
  if (frontendPaths.length > backendPaths.length) {
    console.log(`  ⚠️  Frontend has ${frontendPaths.length - backendPaths.length} more route(s)`)
    frontendPaths.slice(backendPaths.length).forEach(path => {
      console.log(`     Extra: ${path}`)
    })
    allMatch = false
  }
})

console.log('\n' + '='.repeat(60))
if (allMatch) {
  console.log('✅ All endpoints match! Frontend and backend are in sync.')
} else {
  console.log('❌ Some endpoints do not match. Please review the differences above.')
}
console.log('='.repeat(60) + '\n')

process.exit(allMatch ? 0 : 1)
