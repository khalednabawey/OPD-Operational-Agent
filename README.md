---
title: OPD Healthcare Business Intelligence AI Agent
---

# 🏥 OPD Healthcare Business Intelligence AI Agent

This project implements an AI-powered conversational agent designed to provide business intelligence and analytical insights for Outpatient Department (OPD) operations within Andalusia Medical Group. The agent leverages live operational data, a KPI knowledge base, and policy documents to answer user queries, identify performance issues, and suggest actionable recommendations. It also integrates with Microsoft Power Automate for escalation workflows and data request management.

## ✨ Features

- **Conversational AI**: Interact with the agent using natural language to query OPD data.
- **Data Analysis**: Provides insights into various KPIs such as Total Revenue, No. Cases, Patient Retention %, No-Show %, and more.
- **Root Cause Analysis**: Identifies potential drivers and root causes for underperforming KPIs based on a predefined knowledge graph.
- **Actionable Recommendations**: Generates specific, executive-friendly, and operationally actionable recommendations.
- **Personalized Experience**: Integrates with a Dataverse (Microsoft Dynamics 365) HR database to fetch user profiles, enabling personalized responses and scope restrictions based on the user's role and Business Unit (BU).
- **Conversation Memory**: Stores daily conversation summaries and accumulated KPIs, allowing for stateful interactions and context retention.
- **Policy Retrieval-Augmented Generation (RAG)**: Utilizes ChromaDB to retrieve relevant policy documents, providing context for questions related to operational rules, thresholds, and compliance.
- **Power Automate Integration**:
  - **Escalation Workflow**: Automatically triggers Power Automate flows for performance escalations when KPIs are below target or critical thresholds, assigning tasks to relevant managers based on the knowledge base and HR data.
  - **Missing KPI Request**: Allows users to request the addition of new KPIs not found in the existing data or knowledge base.
  - **Missing Entity Request**: Facilitates requests for data related to missing doctors or business units.
- **Interactive Visualizations**: Generates Plotly charts (trends, comparisons, dashboards) to visualize data insights within the Streamlit interface.
- **Flexible Deployment**: Supports local execution or Dockerized deployment using `docker-compose`.

## 🚀 Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web user interface.
- **Groq**: Large Language Model (LLM) for natural language understanding and generation.
- **Pandas**: Data manipulation and analysis.
- **Plotly**: Interactive data visualizations.
- **ChromaDB**: Vector database for storing and retrieving policy embeddings (RAG).
- **Sentence Transformers**: For generating embeddings for RAG.
- **Microsoft Dataverse (Dynamics 365)**: HR database for user authentication and profile retrieval.
- **Microsoft Power Automate**: For triggering external workflows (escalations, requests).
- **SQLAlchemy**: ORM for database interactions (SQLite/PostgreSQL).
- **PostgreSQL / SQLite**: Database for conversation memory.
- **`python-dotenv`**: For managing environment variables.
- **`msal`**: Microsoft Authentication Library for Python (for Dataverse).

## ⚙️ Setup and Installation

### Prerequisites

- Docker (recommended for easy setup)
- Python 3.9+
- `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone <https://github.com/your-username/OPD-Agent-Git.git>
cd OPD-Agent-Git
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory of the project by copying `.env.example` and filling in the required values.

```bash
cp .env.example .env
```

Edit the `.env` file:

```dotenv
# Environment variables for the OPD Healthcare Business Intelligence AI Agent
# This file should be renamed to .env and filled with actual values.
# Do NOT commit your .env file to version control.

# Groq API Key and Model
# Obtain your API key from <https://console.groq.com/keys>
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b # Example: llama3-8b-8192, mixtral-8x7b-32768, qwen/qwen3-32b
GROQ_TEMPERATURE=0.4
GROQ_MAX_TOKENS=1800

# Dataverse (Microsoft Dynamics 365) HR Database Credentials
# These are example values, replace with your actual Dataverse instance details
DATAVERSE_ORG_URL=<https://your_org.crm.dynamics.com>
DATAVERSE_TENANT_ID=your_azure_ad_tenant_id
DATAVERSE_CLIENT_ID=your_azure_ad_app_client_id
DATAVERSE_CLIENT_SECRET=your_azure_ad_app_client_secret

# Optional: Remote ChromaDB and Embedding Server (for RAG)
# If running locally with docker-compose, these will be set by the compose file.
# For local Python-only setup, leave empty or point to local services if running separately.
CHROMA_HOST=http://localhost:8000 # e.g., localhost:8000 or the service name in docker-compose
EMBEDDING_SERVER_URL=http://localhost:7997 # e.g., localhost:7997 or the service name in docker-compose

# Database URL for conversation memory (PostgreSQL or SQLite)
# Example for PostgreSQL: postgresql://user:password@host:port/database_name
# Example for SQLite (local file): sqlite:///opd_agent.db
DATABASE_URL=sqlite:///opd_agent.db

# Power Automate Flow URLs (replace with your actual flow URLs)
POWER_AUTOMATE_URL=<https://your_power_automate_escalation_flow_url>
KPI_REQUEST_FLOW_URL=<https://your_power_automate_kpi_request_flow_url>
ENTITY_REQUEST_FLOW_URL=<https://your_power_automate_entity_request_flow_url>

# Email for KPI request assignee (used in Power Automate flow)
KPI_REQUEST_ASSIGNEE_EMAIL=your_email@example.com
```

**Important**: Ensure your Power Automate flow URLs are correctly configured in the `.env` file. These are critical for the escalation and request features.

### 3. Prepare Data Files

Place your `OPD dataset.xlsx` and `Knowledge base.xlsx` files in the `Dataset/` directory.

### 4. Dockerized Setup (Recommended)

This setup uses `docker-compose` to run the Streamlit app, PostgreSQL database, ChromaDB, and an embedding server.

```bash
docker-compose up --build -d
```

This command will:

1.  Build the `opd-agent` Docker image.
2.  Start a PostgreSQL container (`chat_db`).
3.  Start a ChromaDB server container (`chroma-server`).
4.  Start an embedding server container (`embedding-server`) for RAG.
5.  Run a `policy_ingestor` container to build the ChromaDB collection (this runs once).
6.  Start the Streamlit application (`opd-agent`).

Once all services are up and healthy (this might take a few minutes for the embedding server to download its model), you can access the Streamlit app.

### 5. Local Python-only Setup (Alternative)

If you prefer to run without Docker Compose for the application itself (though ChromaDB and the embedding server can still be run via Docker separately):

1.  **Install Python Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Build ChromaDB for Policy RAG**:

    The `policy_rag.py` script needs to be run once to create the ChromaDB collection on disk. This will download the embedding model (if `EMBEDDING_SERVER_URL` is not set) and embed your policy summaries.

    ```bash
    python policy_rag.py
    ```

    If you are using a remote ChromaDB or embedding server, ensure `CHROMA_HOST` and `EMBEDDING_SERVER_URL` are set in your `.env` file before running this.

3.  **Run the Streamlit Application**:

    ```bash
    streamlit run app.py
    ```

## 🌐 Access the Application

Once the application is running (either via Docker or locally), open your web browser and navigate to:

`http://localhost:8501`

## 📂 Project Structure

- `app.py`: The main Streamlit application file, handling the UI and orchestrating agent interactions.
- `agent_core.py`: Contains the core logic of the AI agent, including LLM integration, data processing, KPI analysis, and tool calling for Power Automate flows.
- `hr_user_db.py`: A client for connecting to the Dataverse HR database to fetch employee records and organizational structure for user profiling.
- `conversation_memory.py`: Manages the storage and retrieval of conversation history and FAQ entries in a database (SQLite or PostgreSQL).
- `policy_rag.py`: Implements the Retrieval-Augmented Generation (RAG) system for policy documents using ChromaDB. It handles embedding and querying policy summaries.
- `Dataset/`: Directory containing the `OPD dataset.xlsx` and `Knowledge base.xlsx` files.
- `chroma_db/`: Directory where the ChromaDB vector store is persisted (created by `policy_rag.py`).
- `.env.example`: A template for environment variables.
- `requirements.txt`: Lists all Python dependencies.
- `docker-compose.yml`: Defines the multi-container Docker application setup.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License.
