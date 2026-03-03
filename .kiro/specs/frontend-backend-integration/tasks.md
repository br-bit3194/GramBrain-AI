# Implementation Plan

- [x] 1. Fix API Client Base Configuration





  - Update API client to use correct base URL with `/api` prefix
  - Configure axios instance with proper defaults
  - Add request/response interceptors for logging
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement Authentication Header Injection




  - Create axios request interceptor to add Authorization header
  - Read token from store/localStorage
  - Handle cases where token is not present
  - _Requirements: 1.5_

- [ ] 3. Implement Centralized Error Handling




  - Create error transformation utility function
  - Add axios response interceptor for error handling
  - Map HTTP status codes to user-friendly messages
  - Handle network errors separately from API errors
  - _Requirements: 1.3, 1.4, 6.2_

- [ ]* 3.1 Write property test for error transformation
  - **Property 4: Error Message Transformation**
  - **Validates: Requirements 1.4, 6.2**


- [ ] 4. Update API Client Endpoint Methods

  - Fix all endpoint paths to match backend routes exactly
  - Update auth endpoints (/api/auth/register, /api/auth/login, /api/auth/me)
  - Update user endpoints (/api/users/*)
  - Update farm endpoints (/api/farms/*)
  - Update query endpoint (/api/query)
  - Update product endpoints (/api/products/*)
  - Update knowledge endpoints (/api/knowledge/*)
  - _Requirements: 1.2_

- [ ]* 4.1 Write property test for API URL construction
  - **Property 1: API URL Construction**
  - **Validates: Requirements 1.1**

- [ ]* 4.2 Write property test for request headers
  - **Property 2: Request Header Inclusion**
  - **Validates: Requirements 1.5**


- [ ] 5. Enhance Zustand Store with Auth State

  - Add accessToken and refreshToken to store
  - Add setTokens action
  - Add logout action that clears all auth data
  - Implement store persistence to localStorage
  - _Requirements: 2.3, 8.3_

- [ ]* 5.1 Write property test for token storage
  - **Property 5: Authentication Token Storage**
  - **Validates: Requirements 2.3**

- [ ]* 5.2 Write property test for logout cleanup
  - **Property 11: Logout Cleanup**
  - **Validates: Requirements 8.3**

- [ ] 6. Create useAuth Hook
  - Implement login function that calls API and updates store
  - Implement register function that calls API and updates store
  - Implement logout function that clears store and storage
  - Add isAuthenticated computed value
  - Add loading and error state management
  - _Requirements: 2.1, 2.2, 2.3, 8.3_

- [ ]* 6.1 Write unit tests for useAuth hook
  - Test login flow with valid credentials
  - Test register flow with valid data
  - Test logout flow
  - Test error handling for invalid credentials
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 7. Implement Login Page
  - Create login form with phone and password fields
  - Add form validation
  - Connect form to useAuth hook
  - Display loading state during login
  - Display error messages
  - Redirect to dashboard on success
  - _Requirements: 2.2_

- [ ] 8. Implement Register Page
  - Create registration form with required fields
  - Add form validation
  - Connect form to useAuth hook
  - Display loading state during registration
  - Display error messages
  - Redirect to dashboard on success
  - _Requirements: 2.1_

- [ ] 9. Create ProtectedRoute Component
  - Check for authentication token
  - Redirect to login if not authenticated
  - Render children if authenticated
  - _Requirements: 2.4, 8.4_

- [ ]* 9.1 Write property test for protected route redirection
  - **Property 6: Protected Route Redirection**
  - **Validates: Requirements 2.4, 8.4**

- [ ] 10. Implement Session Restoration
  - Load tokens from localStorage on app initialization
  - Call /api/auth/me to verify token and get user data
  - Update store with user data if token is valid
  - Clear storage if token is invalid
  - _Requirements: 8.2_

- [ ]* 10.1 Write property test for session restoration
  - **Property 12: Session Restoration**
  - **Validates: Requirements 8.2**

- [ ] 11. Update App Layout with Auth State
  - Show login/register buttons when not authenticated
  - Show user menu and logout when authenticated
  - Update header navigation based on auth state
  - _Requirements: 8.1_

- [ ] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Create useFarm Hook
  - Implement loadFarms function to fetch user's farms
  - Implement selectFarm function to set active farm
  - Implement createFarm function to create new farm
  - Add loading and error state management
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ]* 13.1 Write property test for farm association
  - **Property 7: Farm Association**
  - **Validates: Requirements 3.1**

- [ ] 14. Update Farms Page
  - Display list of user's farms using useFarm hook
  - Add farm selection functionality
  - Show "Create your first farm" prompt when no farms exist
  - Display loading state while fetching farms
  - Display error messages if fetch fails
  - _Requirements: 3.2, 3.5_

- [ ] 15. Implement Farm Creation Form
  - Create form with location, area, soil type fields
  - Add form validation
  - Connect to useFarm hook
  - Display success message on creation
  - Update farms list after creation
  - _Requirements: 3.1_

- [ ] 16. Implement Farm Details View
  - Display selected farm information
  - Show farm location on map (if available)
  - Display farm crops and details
  - Add edit functionality
  - _Requirements: 3.3, 3.4_

- [ ] 17. Create useQuery Hook
  - Implement submitQuery function that calls /api/query
  - Add loading state management
  - Add error state management
  - Store recommendation result
  - Implement clearRecommendation function
  - _Requirements: 4.1, 4.2, 4.4_

- [ ]* 17.1 Write property test for query payload structure
  - **Property 8: Query Payload Structure**
  - **Validates: Requirements 4.1**

- [ ]* 17.2 Write property test for loading state management
  - **Property 9: Loading State Management**
  - **Validates: Requirements 4.2, 6.1**

- [ ] 18. Update Query Page
  - Connect query form to useQuery hook
  - Include farm context in query submission
  - Display loading spinner during processing
  - Show recommendation when received
  - Display confidence score and reasoning chain
  - Add error handling with retry option
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 19. Implement Query History View
  - Fetch user's past recommendations
  - Display recommendations with timestamps
  - Allow viewing full recommendation details
  - Add pagination if needed
  - _Requirements: 4.5_

- [ ] 20. Update Marketplace Page
  - Fetch products using API client
  - Display product grid with details and scores
  - Show loading state while fetching
  - Display error message if fetch fails
  - _Requirements: 5.1_

- [ ] 21. Implement Product Filtering
  - Add filter controls (product type, price, score)
  - Update API call with filter parameters
  - Refresh product list when filters change
  - _Requirements: 5.2_

- [ ]* 21.1 Write property test for filter propagation
  - **Property 10: Product Filter Propagation**
  - **Validates: Requirements 5.2**

- [ ] 22. Implement Product Creation Form
  - Create form for farmers to list products
  - Add form validation
  - Connect to API client
  - Display success message on creation
  - Show Pure Product Score after creation
  - _Requirements: 5.3_

- [ ] 23. Implement Product Details View
  - Display complete product information
  - Show farmer details
  - Display Pure Product Score prominently
  - Add images if available
  - _Requirements: 5.4_

- [ ] 24. Implement Farmer Products View
  - Fetch products for specific farmer
  - Display farmer's product listings
  - Add edit/delete functionality for own products
  - _Requirements: 5.5_

- [ ] 25. Create Loading Spinner Component
  - Design reusable loading spinner
  - Add to all pages with async operations
  - _Requirements: 6.1_

- [ ] 26. Create Error Message Component
  - Design user-friendly error display
  - Support different error types
  - Add retry button where appropriate
  - _Requirements: 6.2, 6.4_

- [ ] 27. Create Success Notification Component
  - Design success toast/notification
  - Auto-dismiss after timeout
  - Support custom messages
  - _Requirements: 6.3_

- [ ] 28. Implement Form Validation Feedback
  - Highlight invalid fields with red borders
  - Display field-specific error messages
  - Prevent submission when validation fails
  - _Requirements: 6.5_

- [ ] 29. Add Retry Logic for Failed Requests
  - Implement exponential backoff
  - Add retry button to error messages
  - Limit maximum retry attempts
  - _Requirements: 6.4_

- [ ] 30. Update TypeScript Types
  - Ensure all API request types match backend
  - Ensure all API response types match backend
  - Add proper type exports
  - Fix any type errors
  - _Requirements: 7.1, 7.2_

- [ ] 31. Implement Navigation State Management
  - Maintain auth state across page navigation
  - Maintain farm context across navigation
  - Update active farm when navigating to farm-specific pages
  - _Requirements: 8.1, 8.5_

- [ ] 32. Add Environment Variable Configuration
  - Create .env.local file with NEXT_PUBLIC_API_URL
  - Update API client to use environment variable
  - Document environment variables in README
  - _Requirements: 1.1_

- [ ] 33. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 34. Write integration tests for authentication flow
  - Test complete register → login → access protected route → logout flow
  - Verify tokens are stored and used correctly
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 8.3_

- [ ]* 35. Write integration tests for farm management flow
  - Test create farm → list farms → select farm → update farm flow
  - Verify farm data persists correctly
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ]* 36. Write integration tests for query flow
  - Test submit query → view recommendation → view history flow
  - Verify recommendations are displayed correctly
  - _Requirements: 4.1, 4.2, 4.3, 4.5_

- [ ]* 37. Write integration tests for marketplace flow
  - Test browse products → apply filters → view product → create listing flow
  - Verify products are fetched and displayed correctly
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ]* 38. Achieve 80% test coverage
  - Run coverage report
  - Add tests for uncovered code paths
  - Focus on critical paths first
  - _Requirements: All_
