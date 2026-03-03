# AI Job Search & Email Outreach Automation (Backend)

An AI-powered platform that automates job searching, HR contact extraction, and personalized email outreach. Powered by LangGraph multi-agent workflows, MCP tool servers, and OpenAI.

## 🚀 Features

- **Multi-Agent Workflow**: Utilizes LangGraph to coordinate complex workflows across multiple specialized agents.
- **Specialized AI Agents**:
  - `CV Agent`: Parses and extracts structured data from candidate resumes (PDF/DOCX).
  - `Search Agent`: Automates internet searches for job openings using Tavily and SerpAPI.
  - `HR Agent`: Scrapes and extracts HR contact information from targeted company websites.
  - `Email Agent`: Drafts context-aware, highly personalized outreach emails.
- **MCP Tool Servers**:
  - `Email Server`: Integrates with SMTP to send out customized emails.
  - `Scrape Server`: Web scraping toolkit for extracting deep company information.
  - `Search Server`: Integrates with external search APIs.
- **FastAPI Core**: High-performance asynchronous API backend serving RESTful endpoints.

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **Workflow Orchestration**: LangGraph, LangChain
- **LLM Engine**: OpenAI API (GPT-4o)
- **Search Providers**: SerpAPI, Tavily
- **Tooling**: Model Context Protocol (MCP) concepts

## ⚙️ Configuration & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kinganupamdutta27/ai-job-search-backend.git
   cd ai-job-search-backend
   ```

2. **Set up the Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file based on `.env.example`:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   LANGSMITH_API_KEY=your_langsmith_key
   SERPAPI_API_KEY=your_serpapi_key
   TAVILY_API_KEY=your_tavily_key
   SMTP_EMAIL=your_email@example.com
   SMTP_PASSWORD=your_app_password
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

## 📚 API Endpoints

- `GET /health` - Detailed system health check & configured API status.
- `POST /cv/...` - Upload and parse resumes.
- `POST /workflow/...` - Trigger and monitor LangGraph automation workflows.
- `POST /email/...` - Manage email outreach templates and campaigns.
- `GET/POST /settings/...` - Manage system configurations and API keys dynamically.

## 📄 License

This project is proprietary and intended for personal/internal use.
