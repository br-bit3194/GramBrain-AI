#!/bin/bash

# GramBrain AI - Run Script
# This script helps you run different components of the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
check_python() {
    print_header "Checking Python Version"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
}

# Install dependencies
install_deps() {
    print_header "Installing Dependencies"
    
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    print_info "Installing packages..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    source venv/bin/activate
    
    if [ "$1" == "coverage" ]; then
        print_info "Running tests with coverage..."
        pytest tests/ --cov=src --cov-report=html
        print_success "Coverage report generated in htmlcov/index.html"
    elif [ "$1" == "verbose" ]; then
        print_info "Running tests (verbose)..."
        pytest tests/ -vv
    else
        print_info "Running tests..."
        pytest tests/ -v
    fi
}

# Run specific test file
run_test_file() {
    print_header "Running Test File: $1"
    
    source venv/bin/activate
    pytest "tests/$1" -v
}

# Start API server
start_api() {
    print_header "Starting API Server"
    
    source venv/bin/activate
    
    print_info "Starting server on http://localhost:8000"
    print_info "Swagger UI available at http://localhost:8000/docs"
    print_info "Press Ctrl+C to stop"
    
    python main.py
}

# Run example
run_example() {
    print_header "Running Example"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import asyncio
from src.system import GramBrainSystem

async def main():
    print("Initializing GramBrain system...")
    system = GramBrainSystem(use_mock_llm=True, use_mock_rag=True)
    await system.initialize()
    
    print("\nAdding knowledge...")
    await system.add_knowledge(
        chunk_id="wheat_irrigation_1",
        content="Wheat requires 450-600mm of water during growing season. Optimal irrigation timing is at tillering and grain filling stages.",
        source="best_practice",
        topic="irrigation",
        crop_type="wheat",
        region="north_india",
    )
    
    print("Processing query...")
    result = system.process_query(
        query_text="Should I irrigate my wheat field?",
        user_id="farmer_001",
        farm_id="farm_001",
        farm_location={"lat": 28.5, "lon": 77.0},
        farm_size_hectares=2.0,
        crop_type="wheat",
        growth_stage="tillering",
        soil_type="loamy",
    )
    
    print("\n" + "="*50)
    print("RECOMMENDATION")
    print("="*50)
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"\nReasoning Chain:")
    for i, step in enumerate(result['reasoning_chain'], 1):
        print(f"  {i}. {step}")
    
    system.shutdown()
    print("\nExample completed successfully!")

asyncio.run(main())
EOF
}

# Show help
show_help() {
    cat << EOF
${BLUE}GramBrain AI - Run Script${NC}

Usage: ./run.sh [command] [options]

Commands:
  ${GREEN}setup${NC}              Install dependencies and setup environment
  ${GREEN}test${NC}               Run all tests
  ${GREEN}test coverage${NC}      Run tests with coverage report
  ${GREEN}test verbose${NC}       Run tests with verbose output
  ${GREEN}test <file>${NC}        Run specific test file (e.g., test_agents.py)
  ${GREEN}api${NC}                Start API server
  ${GREEN}example${NC}            Run example usage
  ${GREEN}help${NC}               Show this help message

Examples:
  ./run.sh setup              # Install dependencies
  ./run.sh test               # Run all tests
  ./run.sh test coverage      # Run tests with coverage
  ./run.sh test test_agents.py # Run specific test file
  ./run.sh api                # Start API server
  ./run.sh example            # Run example

${YELLOW}Quick Start:${NC}
  1. ./run.sh setup           # Install dependencies
  2. ./run.sh test            # Run tests
  3. ./run.sh api             # Start API server
  4. Open http://localhost:8000/docs in browser

${YELLOW}For more information:${NC}
  - README.md - Project overview
  - API.md - API documentation
  - TESTING.md - Testing guide
  - QUICKSTART.md - Quick start guide

EOF
}

# Main script
main() {
    case "$1" in
        setup)
            check_python
            install_deps
            print_success "Setup complete! Run './run.sh api' to start the server"
            ;;
        test)
            check_python
            if [ ! -d "venv" ]; then
                install_deps
            fi
            if [ -z "$2" ]; then
                run_tests
            elif [ "$2" == "coverage" ] || [ "$2" == "verbose" ]; then
                run_tests "$2"
            else
                run_test_file "$2"
            fi
            ;;
        api)
            check_python
            if [ ! -d "venv" ]; then
                install_deps
            fi
            start_api
            ;;
        example)
            check_python
            if [ ! -d "venv" ]; then
                install_deps
            fi
            run_example
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
