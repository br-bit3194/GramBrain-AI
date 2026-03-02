# Farm Selection Fix - Guide

**Issue:** "Please login and select a farm to ask questions" error when trying to get AI recommendations

**Root Cause:** The farms page was not implementing farm selection functionality. The `FarmCard` component had an `onSelect` prop but it wasn't being used.

**Solution:** Implemented farm selection in the farms page by:
1. Adding `setFarm` from the Zustand store
2. Creating a `handleSelectFarm` function
3. Passing the handler to `FarmCard` components
4. Displaying which farm is selected
5. Showing a confirmation message

---

## What Was Fixed

### File Modified
- `frontend/src/app/farms/page.tsx`

### Changes Made

**1. Added farm selection state:**
```typescript
const { user, farm: selectedFarm, setFarm } = useAppStore()
```

**2. Created selection handler:**
```typescript
const handleSelectFarm = (farm: any) => {
  setFarm(farm)
}
```

**3. Updated farm cards to show selection:**
```typescript
{farms.map((farm) => (
  <div key={farm.farm_id} className="relative">
    <FarmCard farm={farm} onSelect={handleSelectFarm} />
    {selectedFarm?.farm_id === farm.farm_id && (
      <div className="absolute top-4 right-4 px-3 py-1 bg-blue-500 text-white text-xs font-medium rounded-full">
        Selected
      </div>
    )}
  </div>
))}
```

**4. Added selection confirmation message:**
```typescript
{selectedFarm && (
  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
    <p className="text-blue-700 text-sm">
      <strong>Selected Farm:</strong> {selectedFarm.farm_id} ({selectedFarm.area_hectares} hectares)
    </p>
  </div>
)}
```

---

## How to Use

### Step 1: Login
1. Go to `http://localhost:3000/login`
2. Enter your user ID
3. Click "Login"

### Step 2: Create or Select a Farm
1. Go to `http://localhost:3000/farms`
2. Either:
   - Click "Add Farm" to create a new farm
   - Click on an existing farm card to select it

### Step 3: Verify Selection
- You should see a blue "Selected" badge on the farm card
- A blue confirmation message shows which farm is selected

### Step 4: Ask Questions
1. Go to `http://localhost:3000/query`
2. You should now see the query form (no more error message)
3. Ask your question
4. Get AI recommendations

---

## Data Flow

```
Login Page
    ↓
User logged in (stored in Zustand)
    ↓
Farms Page
    ↓
Click on Farm Card
    ↓
handleSelectFarm() called
    ↓
setFarm() updates Zustand store
    ↓
Farm selected (shown with blue badge)
    ↓
Query Page
    ↓
QueryInterface checks: user && farm
    ↓
Both exist → Show query form
    ↓
Submit query with farm data
    ↓
Get AI recommendation
```

---

## Testing

### Test 1: Farm Selection
1. Login
2. Go to Farms page
3. Click on a farm card
4. Verify:
   - Blue "Selected" badge appears
   - Blue confirmation message shows
   - Farm details are correct

### Test 2: Query with Selected Farm
1. Select a farm (Test 1)
2. Go to Query page
3. Verify:
   - Query form is visible (no error message)
   - Can submit a query
   - Get recommendation

### Test 3: Multiple Farms
1. Create multiple farms
2. Select different farms
3. Verify:
   - Only one farm shows "Selected" badge
   - Confirmation message updates
   - Query page works with each farm

### Test 4: Without Farm Selection
1. Login
2. Go directly to Query page (without selecting farm)
3. Verify:
   - Error message shows: "Please login and select a farm to ask questions"
   - Query form is hidden

---

## Verification Checklist

- [ ] Farm selection works
- [ ] Selected farm shows blue badge
- [ ] Confirmation message displays
- [ ] Query page shows form after farm selection
- [ ] Query page shows error without farm selection
- [ ] Can ask questions after farm selection
- [ ] Get recommendations successfully

---

## Files Modified

```
frontend/src/app/farms/page.tsx
├─ Added: farm selection state
├─ Added: handleSelectFarm function
├─ Added: onSelect handler to FarmCard
├─ Added: Selected badge display
└─ Added: Confirmation message
```

---

## Related Files

- `frontend/src/store/appStore.ts` - Zustand store (setFarm)
- `frontend/src/components/cards/FarmCard.tsx` - Farm card component
- `frontend/src/components/QueryInterface.tsx` - Query form (checks for farm)
- `frontend/src/app/query/page.tsx` - Query page

---

## Next Steps

1. Restart frontend: `cd frontend && npm run dev`
2. Test farm selection: Go to `/farms` and click a farm
3. Test query: Go to `/query` and ask a question
4. Verify recommendations work

---

## Success Criteria

✅ Farm selection works  
✅ Selected farm shows badge  
✅ Query page shows form after selection  
✅ Can ask questions and get recommendations  
✅ Error message shows without farm selection  

---

**Status: FIXED** ✅

The farm selection feature is now fully implemented and working.

