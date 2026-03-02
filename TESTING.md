# GramBrain AI - Testing Guide

## Overview

GramBrain AI includes comprehensive unit, integration, and API tests covering:

- **Agent Tests**: Individual agent functionality and analysis
- **Orchestrator Tests**: Multi-agent coordination and synthesis
- **Data Model Tests**: Data serialization and validation
- **RAG Tests**: Knowledge retrieval and embedding
- **API Tests**: REST endpoint functionality

## Running Tests

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run Specific Test File

```bash
pytest tests/test_agents.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_agents.py::TestWeatherAgent -v
```

### Run Specific Test

```bash
pytest tests/test_agents.py::TestWeatherAgent::test_weather_agent_initialization -v
```

### Run Only Async Tests

```bash
pytest tests/ -v -m asyncio
```

### Run with Markers

```bash
# Run only unit tests
pytest tests/ -v -m unit

# Run only integration tests
pytest tests/ -v -m integration
```

## Test Structure

### Unit Tests

Located in `tests/test_*.py` files:

- **test_agents.py**: Individual agent functionality
  - Weather Agent analysis
  - Soil health calculations
  - Pest risk assessment
  - Yield predictions
  - Market analysis
  - Sustainability metrics
  - Pure Product Score calculation
  - Farmer interaction processing
  - Village data aggregation

- **test_orchestrator.py**: Orchestrator coordination
  - Agent selection based on intent
  - Multi-agent synthesis
  - Data source collection
  - Fallback mechanisms

- **test_data_models.py**: Data model validation
  - User model serialization
  - Farm model creation
  - Crop cycle tracking
  - Soil health data
  - Weather data
  - Product listings
  - Recommendations

- **test_rag.py**: RAG pipeline
  - Vector database operations
  - Knowledge chunk management
  - Semantic search
  - Metadata filtering
  - Context formatting

- **test_api.py**: REST API endpoints
  - User management
  - Farm operations
  - Query processing
  - Product marketplace
  - Knowledge management
  - Health checks

## Test Examples

### Testing an Agent

```python
@pytest.mark.asyncio
async def test_weather_agent_analyze(self, query, context):
    """Test weather agent analysis."""
    agent = WeatherAgent()
    output = await agent.analyze(query, context)
    
    assert output.agent_name == "weather_agent"
    assert output.recommendation is not None
    assert 0 <= output.confidence <= 1
    assert len(output.data_sources) > 0
```

### Testing Data Models

```python
def test_farm_creation(self):
    """Test farm creation."""
    farm = Farm(
        farm_id="farm_001",
        owner_id="user_001",
        location={"lat": 28.5, "lon": 77.0},
        area_hectares=2.5,
        soil_type="loamy",
        irrigation_type=IrrigationType.DRIP,
    )
    
    assert farm.farm_id == "farm_001"
    assert farm.area_hectares == 2.5
```

### Testing API Endpoints

```python
def test_process_query(self, client):
    """Test processing a query."""
    response = client.post(
        "/api/v1/query",
        params={
            "user_id": "farmer_001",
            "query_text": "Should I irrigate?",
            "crop_type": "wheat",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
```

## Test Coverage

Current test coverage includes:

- **Agents**: 11 specialized agents with analysis tests
- **Orchestrator**: Agent coordination and synthesis
- **Data Models**: 10+ data model classes
- **RAG Pipeline**: Vector DB, embeddings, retrieval
- **API**: 20+ REST endpoints

Target coverage: **>80%** of core functionality

## Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src
```

## Performance Testing

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

### Benchmark Tests

```bash
pytest tests/ --benchmark-only
```

## Debugging Tests

### Verbose Output

```bash
pytest tests/ -vv
```

### Show Print Statements

```bash
pytest tests/ -s
```

### Drop into Debugger on Failure

```bash
pytest tests/ --pdb
```

### Show Local Variables on Failure

```bash
pytest tests/ -l
```

## Mocking and Fixtures

### Using Fixtures

```python
@pytest.fixture
def context():
    """Create a test user context."""
    return UserContext(
        user_id="farmer_001",
        farm_id="farm_001",
        crop_type="wheat",
    )

def test_something(self, context):
    """Test using fixture."""
    assert context.crop_type == "wheat"
```

### Mocking External Services

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_with_mock():
    """Test with mocked LLM."""
    with patch('src.llm.BedrockClient.invoke') as mock_invoke:
        mock_invoke.return_value = "Mocked response"
        # Test code here
```

## Best Practices

1. **Use Fixtures**: Reuse common test data with fixtures
2. **Test One Thing**: Each test should verify one behavior
3. **Clear Names**: Use descriptive test names
4. **Async Tests**: Mark async tests with `@pytest.mark.asyncio`
5. **Assertions**: Use specific assertions, not generic `assert True`
6. **Cleanup**: Use fixtures for setup/teardown
7. **Isolation**: Tests should not depend on each other
8. **Mocking**: Mock external services and APIs

## Troubleshooting

### Tests Hang

- Check for infinite loops in agent logic
- Verify async/await usage is correct
- Check timeout settings in pytest.ini

### Import Errors

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Async Test Issues

```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check asyncio_mode in pytest.ini
```

## Adding New Tests

1. Create test file in `tests/` directory
2. Name file `test_*.py`
3. Create test class `Test*`
4. Create test methods `test_*`
5. Use fixtures for common setup
6. Run tests: `pytest tests/test_new.py -v`

Example:

```python
# tests/test_new_feature.py
import pytest

class TestNewFeature:
    """Tests for new feature."""
    
    def test_feature_works(self):
        """Test that feature works."""
        result = new_feature()
        assert result is not None
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
