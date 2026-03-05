#!/bin/bash

echo "🌾 GramBrain Setup Script"
echo "=========================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f "backend/.env" ]; then
    echo ""
    echo "⚠️  Warning: backend/.env file not found!"
    echo "Please create backend/.env with your AWS credentials:"
    echo ""
    echo "AWS_REGION=us-east-1"
    echo "AWS_ACCESS_KEY_ID=your_access_key"
    echo "AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0"
    echo "WEATHER_API_KEY=your_openweathermap_key"
    echo ""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
