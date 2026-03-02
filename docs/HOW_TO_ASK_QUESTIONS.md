# How to Ask Questions - Step-by-Step Guide

**Problem:** "Please login and select a farm to ask questions" error

**Solution:** Follow these steps to properly login, select a farm, and ask questions.

---

## Step 1: Start the System

### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

---

## Step 2: Register or Login

### Option A: Register (First Time)
1. Open: `http://localhost:3000`
2. Click "Register" in the header
3. Fill in the form:
   - **Phone:** +91 98765 43210
   - **Name:** Your Name
   - **Language:** English
   - **Role:** Farmer
4. Click "Register"
5. You'll be redirected to Dashboard

### Option B: Login (Existing User)
1. Open: `http://localhost:3000`
2. Click "Login" in the header
3. Enter your user ID (from registration)
4. Click "Login"
5. You'll be redirected to Dashboard

---

## Step 3: Create or Select a Farm

### Option A: Create a New Farm
1. Go to: `http://localhost:3000/farms`
2. Click "Add Farm" button
3. Fill in the form:
   - **Latitude:** 28.7041
   - **Longitude:** 77.1025
   - **Area (hectares):** 5.5
   - **Soil Type:** loamy
   - **Irrigation Type:** drip
4. Click "Create Farm"
5. Your farm is created and automatically selected

### Option B: Select an Existing Farm
1. Go to: `http://localhost:3000/farms`
2. Click on any farm card
3. You'll see a blue "Selected" badge appear
4. A blue confirmation message shows the selected farm

---

## Step 4: Verify Farm Selection

After selecting a farm, you should see:
- ✅ Blue "Selected" badge on the farm card
- ✅ Blue confirmation message: "Selected Farm: [farm_id] ([area] hectares)"

---

## Step 5: Ask Questions

1. Go to: `http://localhost:3000/query`
2. You should now see the query form (no error message)
3. Fill in your question:
   - **Your Question:** "Should I irrigate my wheat field today?"
   - **Crop Type (Optional):** Wheat
   - **Growth Stage (Optional):** Vegetative
4. Click "Get Recommendation"
5. Wait for the AI recommendation

---

## Step 6: View Recommendation

After submitting your question, you'll see:
- ✅ Recommendation text
- ✅ Confidence score (with progress bar)
- ✅ Reasoning chain (how we arrived at the answer)
- ✅ Agents involved (which AI agents contributed)

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Start System                                        │
│ - Backend on port 8000                                      │
│ - Frontend on port 3000                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Register/Login                                      │
│ - Go to http://localhost:3000                               │
│ - Register or Login                                         │
│ - User stored in Zustand store                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Create/Select Farm                                  │
│ - Go to http://localhost:3000/farms                         │
│ - Create new farm OR click existing farm                    │
│ - Farm stored in Zustand store                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Verify Selection                                    │
│ - See blue "Selected" badge                                 │
│ - See confirmation message                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Ask Questions                                       │
│ - Go to http://localhost:3000/query                         │
│ - Query form is now visible                                 │
│ - Fill in your question                                     │
│ - Click "Get Recommendation"                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: View Recommendation                                 │
│ - See AI recommendation                                     │
│ - See confidence score                                      │
│ - See reasoning chain                                       │
│ - See agents involved                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Error: "Please login and select a farm to ask questions"

**Cause:** Farm is not selected in the Zustand store

**Solution:**
1. Go to `/farms`
2. Click on a farm card to select it
3. Verify blue "Selected" badge appears
4. Go back to `/query`
5. Query form should now be visible

### Error: "Please login"

**Cause:** User is not logged in

**Solution:**
1. Go to `/login` or `/register`
2. Login or register
3. You'll be redirected to dashboard
4. Go to `/farms` to select a farm
5. Go to `/query` to ask questions

### Error: "No farms yet"

**Cause:** No farms created

**Solution:**
1. Click "Add Farm" button
2. Fill in farm details
3. Click "Create Farm"
4. Farm is automatically selected
5. Go to `/query` to ask questions

### Query form not showing

**Cause:** User or farm not selected

**Solution:**
1. Check browser console (F12)
2. Go to `/farms`
3. Click on a farm to select it
4. Verify blue badge appears
5. Go to `/query`
6. Query form should now be visible

---

## Quick Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Registered or logged in
- [ ] Created or selected a farm
- [ ] See blue "Selected" badge
- [ ] See confirmation message
- [ ] Query form visible at `/query`
- [ ] Can submit a question
- [ ] Get recommendation

---

## Example Questions

Try asking these questions:

1. **Irrigation:** "Should I irrigate my wheat field today?"
2. **Pest Management:** "How do I prevent pest damage in my rice crop?"
3. **Soil:** "What's the best way to improve my soil quality?"
4. **Weather:** "How will the weather affect my crop this week?"
5. **Yield:** "How can I increase my crop yield?"
6. **Sustainability:** "What sustainable farming practices should I use?"

---

## Expected Results

### Successful Query
```
✅ Query submitted
✅ Processing... (shows spinner)
✅ Recommendation received
✅ Shows:
   - Recommendation text
   - Confidence score
   - Reasoning chain
   - Agents involved
```

### Failed Query
```
❌ Error message displayed
❌ Check:
   - Farm is selected
   - User is logged in
   - Backend is running
   - Network connection
```

---

## Next Steps

1. **Follow the steps above** to ask your first question
2. **Read FARM_SELECTION_FIX.md** for technical details
3. **Read TESTING_GUIDE.md** for comprehensive testing
4. **Read SYSTEM_ARCHITECTURE.md** for system overview

---

**Status: READY TO USE** ✅

Follow the steps above to start asking questions!

