# GramBrain AI - Multi-Agent Agricultural Intelligence Platform

A next-generation, cloud-native, multi-agent AI platform designed to serve as "The AI Brain for Every Village in Bharat". GramBrain AI uses collaborative AI agents powered by Large Language Model (LLM) reasoning and Retrieval-Augmented Generation (RAG) to deliver real-time, explainable, and scalable agricultural intelligence.

## Features

- **Multi-Agent Architecture**: 12 specialized AI agents collaborating through a master orchestrator
- **LLM-Powered Reasoning**: AWS Bedrock integration with Claude, Titan, and Llama models
- **RAG-Based Knowledge Retrieval**: Semantic search for contextual agricultural knowledge
- **Explainable AI**: Transparent reasoning chains for every recommendation
- **Farmer Intelligence**: Personalized crop advisory, weather-aware irrigation, pest detection
- **Village Intelligence**: Aggregated insights for collective decision-making
- **Sustainability Tracking**: Environmental impact metrics and eco-friendly recommendations
- **Rural-to-Urban Marketplace**: Direct farmer-to-consumer connections with Pure Product Scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                           │
│  React Native Mobile App │ Next.js Web Dashboard │ Voice UI     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      API Gateway Layer                           │
│  Amazon API Gateway │ WebSocket API │ REST API │ GraphQL        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                Multi-Agent Intelligence Layer                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Orchestrator Agent (Master)                 │   │
│  │  - Coordinates all agents                                │   │
│  │  - Resolves conflicts using LLM                          │   │
│  │  - Synthesizes final recommendations                     │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  Weather │ Soil │ Crop Advisory │ Pest │ Irrigation │ Yield    │
│  Market │ Sustainability │ Marketplace │ Farmer Interaction    │
│  Village Intelligence                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    LLM & RAG Layer                               │
│  AWS Bedrock (Claude, Titan, Llama) │ Vector Database          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Data Layer                                  │
│  RDS PostgreSQL │ DynamoDB │ Timestream │ S3 Data Lake         │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- AWS CLI configured with credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/grambrain/grambrain-ai.git
cd grambrain-ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

### Usage

```python
import asyncio
from src.system import GramBrainSystem

async def main():
    # Initialize system
    system = GramBrainSystem(aws_region="us-east-1")
    await system.initialize()
    
    # Process a query
    result = system.process_query(
        query_text="Should I irrigate my wheat field today?",
        user_id="farmer_001",
        farm_id="farm_001",
        farm_location={"lat": 28.5, "lon": 77.0},
        farm_size_hectares=2.0,
        crop_type="wheat",
        growth_stage="tillering",
        soil_type="loamy",
    )
    
    print("Recommendation:", result["recommendation"])
    print("Confidence:", result["confidence"])
    
    system.shutdown()

asyncio.run(main())
```

## Project Structure

```
grambrain-ai/
├── src/
│   ├── core/                 # Agent framework and orchestrator
│   ├── agents/               # Specialized AI agents
│   ├── llm/                  # LLM integration (AWS Bedrock)
│   ├── rag/                  # RAG pipeline and knowledge retrieval
│   ├── data/                 # Data models and database layer
│   ├── api/                  # REST API endpoints
│   └── utils/                # Utilities and helpers
├── tests/                    # Unit and integration tests
├── config/                   # Configuration files
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Agents

### Implemented

- **Weather Agent**: Analyzes weather forecasts and provides irrigation recommendations
- **Soil Agent**: Analyzes soil health and recommends amendments

### In Development

- **Crop Advisory Agent**: Crop-specific guidance
- **Pest & Disease Agent**: Pest detection and treatment
- **Irrigation Optimization Agent**: Water usage optimization
- **Yield Prediction Agent**: Harvest forecasting
- **Market Intelligence Agent**: Price predictions and market insights
- **Sustainability Agent**: Environmental impact tracking
- **Marketplace Agent**: Product listings and Pure Product Scores
- **Farmer Interaction Agent**: Multilingual voice interface
- **Village Intelligence Agent**: Collective insights

## Data Models

### Core Entities

- **User**: Farmer, village leader, policymaker, or consumer
- **Farm**: Farm location, size, soil type, crops
- **CropCycle**: Planting, growth, harvest data
- **SoilHealthData**: NPK levels, pH, organic carbon
- **WeatherData**: Temperature, rainfall, forecasts
- **AgentOutput**: Standardized agent analysis and recommendations
- **Recommendation**: Final synthesized recommendation with reasoning
- **Product**: Marketplace product listings with Pure Product Scores

## LLM Integration

GramBrain uses AWS Bedrock for LLM inference:

- **Claude 3 Sonnet**: Primary model for reasoning and synthesis
- **Claude 3 Haiku**: Lightweight model for simple tasks
- **Titan Text**: Alternative model for cost optimization
- **Llama 2 70B**: Open-source alternative

## RAG (Retrieval-Augmented Generation)

Knowledge retrieval system for grounding LLM responses:

- **Vector Database**: In-memory (dev), OpenSearch (production)
- **Embeddings**: AWS Bedrock Titan Embeddings
- **Knowledge Sources**: Research papers, best practices, case studies, government guidelines

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

### AWS Configuration

Ensure your AWS credentials are configured:
```bash
aws configure
```

## Development

### Adding a New Agent

1. Create agent class in `src/agents/`:
```python
from src.core import Agent, AgentOutput, Query, UserContext

class MyAgent(Agent):
    def __init__(self):
        super().__init__("my_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        # Implementation
        pass
```

2. Register in `src/system.py`:
```python
self.registry.register_agent_class("my_agent", MyAgent)
```

### Adding Knowledge to RAG

```python
await system.add_knowledge(
    chunk_id="unique_id",
    content="Knowledge content",
    source="best_practice",
    topic="irrigation",
    crop_type="wheat",
    region="north_india",
)
```

## Performance Targets

- **Recommendation Latency**: < 3 seconds for 95% of queries
- **LLM Accuracy**: 85%+ for yield predictions within ±15%
- **Market Price Forecast**: 75%+ directional accuracy for 7-day forecasts
- **Pest Detection**: 90%+ accuracy from crop images
- **System Availability**: 99.9% uptime

## Correctness Properties

The system implements 30 formal correctness properties covering:

- Multi-agent coordination and communication
- LLM integration and fallback mechanisms
- RAG knowledge retrieval and attribution
- Recommendation explainability
- Data integrity and privacy
- Marketplace features and traceability

See `design.md` for complete property specifications.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or suggestions:
- GitHub Issues: https://github.com/grambrain/grambrain-ai/issues
- Email: support@grambrain.ai

## Roadmap

- Phase 1: Core multi-agent framework (current)
- Phase 2: Mobile app and voice interface
- Phase 3: Marketplace and consumer features
- Phase 4: Advanced analytics and policymaker dashboard
- Phase 5: IoT sensor integration and real-time data pipelines

## Acknowledgments

Built with support from agricultural experts, farmers, and the rural development community.
