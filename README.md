# IRAM - Instagram Research Agent MCP

A comprehensive, autonomous AI agent system for researching, downloading, scraping, fetching, and analyzing Instagram content. IRAM is built as an MCP (Model Context Protocol) server with advanced evasion strategies and multi-modal analysis capabilities.

## 🚀 Features

### Core Capabilities
- **Autonomous Agent**: LangChain ReAct agent that decomposes high-level tasks
- **Multi-Modal Analysis**: NLP for text, computer vision for images/videos
- **Smart Evasion**: ML-based anti-ban detection and adaptive request patterns
- **Comprehensive Scraping**: Profile, posts, stories, followers, and hashtags
- **MCP Integration**: JSON-RPC over stdio/WebSockets for tool integration

### Analysis Features
- Sentiment analysis using state-of-the-art transformers
- Topic modeling with BERTopic
- Engagement pattern analysis
- Hashtag trend research
- Account comparison and benchmarking

### Technical Features
- FastAPI-based MCP server
- Docker containerization
- Playwright browser automation fallback
- Instagram private API integration (Instagrapi)
- Firecrawl v2.2.0 integration for advanced web scraping
- Proxy rotation and session management
- Comprehensive error handling and logging

## 📋 Prerequisites

- Python 3.11+
- Docker (optional)
- Instagram account credentials (for private content)
- OpenAI API key
- Firecrawl API key (for advanced web scraping and search)

## 🛠 Installation

### Local Setup

1. **Clone and setup**
```bash
git clone <repository-url>
cd iram-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Test installation**
```bash
python main.py test
```

### Docker Setup

```bash
docker build -t iram-agent:latest .
docker run -p 8000:8000 --env-file .env iram-agent:latest
```

## 🔧 Configuration

### Required Environment Variables

```bash
# Instagram Credentials
INSTAGRAM_USERNAME=your_instagram_username
INSTAGRAM_PASSWORD=your_instagram_password

# AI/LLM API Keys
OPENAI_API_KEY=your_openai_api_key

# Optional: Facebook/Instagram Graph API
FACEBOOK_APP_ID=your_facebook_app_id
FACEBOOK_APP_SECRET=your_facebook_app_secret
INSTAGRAM_ACCESS_TOKEN=your_long_lived_access_token
INSTAGRAM_BUSINESS_ACCOUNT_ID=your_instagram_business_account_id

# Web Scraping API (Firecrawl v2.2.0)
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## 🎯 Usage

### CLI Interface

```bash
# Start MCP server
python main.py server

# Run specific analysis tasks
python main.py run --task "analyze-account" --username "example_user"
python main.py run --task "profile" --username "example_user" --output results.json

# Custom analysis
python main.py run --task "Compare engagement between @user1 and @user2"

# Show configuration
python main.py config
```

### MCP Server Endpoints

Once running, the server provides these endpoints:

- `POST /execute_task` - Execute high-level tasks
- `POST /fetch_profile` - Get profile information
- `POST /scrape_content` - Scrape specific content
- `POST /analyze_content` - Analyze content
- `POST /analyze_account_trends` - Account trend analysis
- `POST /compare_accounts` - Compare multiple accounts
- `GET /search_users` - Search for users
- `GET /health` - Health check

### Example API Usage

```python
import requests

# Analyze account trends
response = requests.post('http://localhost:8000/analyze_account_trends', 
                        params={'username': 'example_user', 'days': 30})
print(response.json())

# Compare accounts
response = requests.post('http://localhost:8000/compare_accounts',
                        json={'usernames': ['user1', 'user2', 'user3']})
print(response.json())
```

### Agent Task Examples

IRAM can handle natural language tasks like:

- "Analyze @username's engagement trends over the last 30 days"
- "Compare content strategies between @brand1 and @brand2"  
- "Research hashtag #photography trends and top performing posts"
- "Find similar accounts to @influencer based on content themes"
- "Analyze sentiment of comments on @brand's recent posts"

## 🏗 Architecture

### Core Components

1. **Agent Orchestrator** (`agent_orchestrator.py`)
   - LangChain ReAct agent
   - Task decomposition and execution
   - Tool integration

2. **MCP Server** (`mcp_server.py`)
   - FastAPI-based REST API
   - Background task processing
   - Error handling and validation

3. **Scraping Module** (`scraping_module.py`)
   - Instagrapi for private API access
   - Playwright for browser automation
   - Content extraction and processing

4. **Analysis Module** (`analysis_module.py`)
   - Sentiment analysis with transformers
   - Topic modeling with BERTopic
   - Engagement pattern analysis
   - Multi-modal content understanding

5. **Evasion Manager** (`evasion_manager.py`)
   - ML-based risk assessment
   - Adaptive request timing
   - Proxy rotation
   - Error pattern analysis

## 🚀 Deployment

### Railway Deployment

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Deploy**
```bash
railway login
railway init --name iram-agent
railway up
```

3. **Set environment variables**
```bash
railway variables set OPENAI_API_KEY=your_key
railway variables set INSTAGRAM_USERNAME=your_username
# ... set other required variables
```

### Docker Deployment

```bash
docker build -t iram-agent:latest .
docker run -d -p 8000:8000 --env-file .env --name iram-agent iram-agent:latest
```

## 📊 Analysis Capabilities

### Sentiment Analysis
- Real-time sentiment scoring
- Emotional trend analysis
- Multi-language support
- Confidence scoring

### Topic Modeling
- Automatic topic discovery
- Keyword extraction
- Content theme evolution
- Trend identification

### Engagement Analysis
- Like/comment patterns
- Optimal posting times
- Audience interaction metrics
- Content performance ranking

### Account Comparison
- Cross-account benchmarking
- Strategy analysis
- Growth pattern comparison
- Competitive intelligence

## 🔒 Security & Ethics

- **Public-first approach**: Prioritizes publicly available data
- **Rate limiting**: Intelligent request spacing to avoid detection
- **Session management**: Secure credential handling
- **Audit logging**: Comprehensive operation tracking
- **Consent checking**: Validation for private content access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

Comprehensive engineering guidelines live in [`AGENTS.md`](AGENTS.md); review them before opening a pull request.

## 📝 License

This project is for educational and research purposes. Ensure compliance with Instagram's Terms of Service and applicable laws.

## 🆘 Support

- Check the test suite: `python main.py test`
- Review logs for debugging
- Verify environment configuration
- Check Instagram API limits and restrictions

## 🔄 Updates

IRAM is actively developed with regular updates for:
- New analysis features
- Improved evasion strategies
- Enhanced agent capabilities
- Platform API changes

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. Users are responsible for complying with Instagram's Terms of Service and applicable laws. Always respect privacy and ethical guidelines when analyzing social media content.
