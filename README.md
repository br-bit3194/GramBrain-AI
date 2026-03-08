# GramBrain AI 🌾 - Providing farmers with expert help on demand

GramBrain AI  is a voice-first, multilingual AI assistant for small-scale farmers. Powered by AWS technologies and the AWS Bedrock model, it delivers crop disease diagnosis, real-time market insights, and government scheme guidance via natural voice in local languages, working even in low-network regions.

---

## 🎯 **Major Challenges Faced by Indian Farmers**

Indian farmers face multiple interconnected challenges that affect their productivity and income:
- **Limited access to expert agricultural advice**
- **Language barriers with technology solutions** 
- **Lack of real-time market and Weather information**
- **Difficulty in identifying crop diseases early**
- **Complex government scheme navigation**
- **Poor internet connectivity in rural areas**
- **Need for natural voice interaction in local languages**

---

## 💡 **Our Solution**

GramBrain AI addresses these challenges through:
- **AI-powered agricultural expertise** available 24/7
- **Voice-first interface** in multiple Indian languages
- **Real-time data integration** from government sources
- **Offline capabilities** for low-connectivity areas
- **Simplified access** to government schemes,weather and market data

---

## ✨ **Key Features**

<div align="center">

| Feature | Description | Technology |
|---------|-------------|------------|
| 🌱 **Crop Disease Diagnosis** | Vision AI–powered photo-based detection with localized remedies | Google Gemini Vision |
| 🌤️ **Real-Time Weather Information** | Real-Time and Weather Forecasting for 7 days | OpenWeatherMap API |
| 📈 **Real-Time Market Intelligence** | Live mandi prices & crop trends from government sources | AgMarkNet API |
| 🏛️ **Government Scheme Navigator** | Eligibility checks & simplified explanations | Data.gov.in APIs |
| 🗣️ **Voice-First & Multilingual** | Supports multiple languages as enabled by Google ADK | Speech Recognition |
| 🎙️ **Premium Voice Synthesis** | High-quality Hindi voice responses with ElevenLabs AI | ElevenLabs TTS |
| 📶 **Offline Support** | Cached responses via Gemini model for low-network zones | PWA + Service Workers |

</div>

### **🎬 Feature Showcase**

#### 🌱 **Govt Scheme**


#### 🗣️ **Weather and Irrigation**


#### 📊 **Market Intelligence**


---

## 🏗️ **Tech Stack**

<div align="center">

| Layer | Technology | Purpose |
|-------|------------|---------|
| **🎨 Frontend** | Progressive Web App (PWA) | Mobile-first, offline-capable interface |
| **⚡ Backend** | FastAPI, Google Cloud Run, Cloud Storage | High-performance API with cloud scaling |
| **🧠 AI & ML** | Google ADK, Gemini, Vision AI | Intelligent conversations and image analysis |
| **🎙️ Voice AI** | ElevenLabs Text-to-Speech | Premium voice synthesis |
| **🔗 Integrations** | AgMarkNet, eNAM, Weather APIs, Data.gov | Real-time agricultural data |

</div>

---

## 🔑 **API Configuration**

| Service | Environment Variable | Required | Get API Key | Purpose |
|---------|---------------------|----------|-------------|---------|
| 🤖 **AWS Bedrock ** | `AWS_ACCESS_KEY_ID` | ✅ **Required**  | AI conversations & vision |
| 🌤️ **Weather API** | `WEATHER_API_KEY` | ✅ **Required** | [Get Key →](https://openweathermap.org/api) | Weather forecasting |
| 🎙️ **ElevenLabs** | `ELEVENLABS_API_KEY` | ⭐ **Recommended** | [Get Key →](https://elevenlabs.io) | Premium voice synthesis |
| 📊 **Data.gov.in** | `MANDI_API_KEY` | ⚠️ *Optional* | [Get Key →](https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi) | Government market data |

### **Environment Setup**
```bash
# Create .env file
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_kry
WEATHER_API_KEY=your_openweathermap_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here  # For premium voice
MANDI_API_KEY=your_data_gov_api_key  # Optional
```
### **🎙️ ElevenLabs Setup**
1. **Sign up** at [ElevenLabs.io](https://elevenlabs.io)
2. **Get API key** from your profile settings
3. **Free tier**: 10,000 characters/month
4. **Paid tiers**: Higher limits + more voice options

---

## 🛠️ **Getting Started**

### **⚡ Quick Start**
```bash
# Clone the repository
cd GramBrain

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (add your API keys)
cp .env.example .env

# Run locally
uvicorn app.main:app --reload
```

---
### **🎙️ Testing Voice Features**
1. Open `http://localhost:8000`
2. Look for **🎙️ Voice ON** toggle in chat header
3. Send a message: "मौसम कैसा है?"
4. Listen to high-quality Hindi voice response!

### **🔧 Voice Configuration**
Customize voice settings in `app/google_adk_integration/services/elevenlabs_voice_service.py`:

```python
# Adjust voice parameters
self.voice_settings = {
    "stability": 0.5,        # 0-1: Lower = more expressive
    "similarity_boost": 0.75, # 0-1: Higher = more like original
    "style": 0.5,            # 0-1: Style strength
    "use_speaker_boost": True # Better quality
}
```

---

## 📁 **Project Structure**

```
GramBrain-AI/
├── app/
│   ├── google_adk_integration/
│   │   ├── agents/              # AI agents for different domains
│   │   ├── tools/               # AI tool functions
│   │   ├── services/            
│   │   │   ├── elevenlabs_voice_service.py  # 🆕 Voice synthesis
│   │   │   └── ...              # Other business logic
│   │   └── farmbot_service.py   # Main AI service (updated with voice)
│   ├── templates/
│   │   └── home.html           # Frontend interface (voice-enabled)
│   ├── main.py                 # FastAPI application (voice endpoints)
│   └── websocket_conn.py       # WebSocket connections
├── requirements.txt            # Updated with voice dependencies
└── README.md                   # This file
```

---

## 🌐 **Useful Resources**

### **📚 Documentation**
- [Google ADK](https://google.github.io/adk-docs/streaming/custom-streaming-ws/#websocket-handling) - Agent Development Kit
- [ElevenLabs API](https://docs.elevenlabs.io/) - Voice synthesis documentation
- [Speech Synthesis Guide](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis) - Browser TTS fallback

### **🎙️ Voice AI Resources**
- [ElevenLabs Voice Library](https://elevenlabs.io/voice-library) - Explore available voices
- [Voice Cloning Guide](https://elevenlabs.io/docs/voice-cloning) - Custom voice creation
- [Audio Quality Tips](https://elevenlabs.io/docs/audio-quality) - Optimization guide

### **🌾 Agricultural Data Sources**
- [AgMarkNet Portal](https://agmarknet.gov.in/) - Government market prices
- [eNAM Platform](https://enam.gov.in/) - National agriculture market
- [Data.gov.in](https://data.gov.in/) - Open government data
- [Weather API](https://openweathermap.org/) - Weather forecasting service

### **🛠️ Development Resources**
- [Google Cloud Run](https://cloud.google.com/run) - Deployment platform
- [WebSocket Guide](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket) - Real-time communication
- [Speech Recognition API](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition) - Voice interface

---

<div align="center">

## 🌾 **Built with ❤️ for Indian Farmers**


*Transforming farming through artificial intelligence*

</div>
