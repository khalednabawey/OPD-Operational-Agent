import warnings
import streamlit as st
import re
from dotenv import load_dotenv
from agent_core import AgentConfig, GroqLLM, OPDDataModel, OPDHealthcareAgent, trigger_flow, trigger_kpi_request_flow, trigger_entity_request_flow
from hr_user_db import EmployeeHRDB
import faulthandler
from pathlib import Path
import os
import sys
import requests
from typing import Optional

# Suppress noisy transformers deprecation warnings
import logging
logging.getLogger("transformers.utils.import_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

_CRASH_LOG = open(Path(__file__).with_name(
    "streamlit_crash.log"), "a", encoding="utf-8")
faulthandler.enable(file=_CRASH_LOG, all_threads=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(".env")

APP_TITLE = "OPD Healthcare Business Intelligence AI Agent"
OPD_PATH = "Dataset/OPD dataset.xlsx"
KB_PATH = "Dataset/Knowledge base.xlsx"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
POWER_AUTOMATE_URL = "https://45f38c914191ef2a95e1ca23996f6e.4d.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/4fc6a51147f440f1a5f8acf604891ff4/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=NCU3OjKuD1tM9NIgIJK1ylEcc75Phr9CLoc-lfjwqkk"
KPI_REQUEST_FLOW_URL = "https://45f38c914191ef2a95e1ca23996f6e.4d.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/ecc6e042141e47478fcafedce88a6508/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=KCXOBHbM_zMJXr3FxInywrIrG28OurPhBiuUkNMdUlE"
ENTITY_REQUEST_FLOW_URL = "https://45f38c914191ef2a95e1ca23996f6e.4d.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/e4f5e68f14f640d08bb94daada7fcab9/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=2gqGcLj2rDKU2u-hKcSfdZEpgj_H1orSuadYIz5fkAM"

# ChromaDB persistent folder (must exist after running policy_rag.py)
CHROMA_DB_FOLDER = Path(__file__).parent / "chroma_db"

st.set_page_config(page_title="OPD AI Agent", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner=False)
def load_agent() -> OPDHealthcareAgent:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to .env or Streamlit secrets.")

    # Only warn about missing local folder if we aren't using a remote host
    chroma_host = os.getenv("CHROMA_HOST")
    if not CHROMA_DB_FOLDER.exists() and not chroma_host:
        st.warning(
            f"ChromaDB folder '{CHROMA_DB_FOLDER}' not found.\n"
            "RAG will be disabled. To enable policy retrieval, run `python policy_rag.py` first."
        )

    opd_path = Path(OPD_PATH)
    kb_path = Path(KB_PATH)
    if not opd_path.exists():
        raise FileNotFoundError(f"Missing OPD dataset: {opd_path}")
    if not kb_path.exists():
        raise FileNotFoundError(f"Missing knowledge base: {kb_path}")

    data_model = OPDDataModel(opd_path, kb_path)
    llm = GroqLLM(
        AgentConfig(
            groq_api_key=GROQ_API_KEY,
            groq_model=GROQ_MODEL,
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.4")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1800")),
        )
    )

    # Initialize HR Database Client
    hr_db = None
    try:
        hr_db = EmployeeHRDB(
            org_url=os.getenv("DATAVERSE_ORG_URL",
                              "https://org2f45e702.crm4.dynamics.com"),
            tenant_id=os.getenv("DATAVERSE_TENANT_ID",
                                "c515f6b1-812f-4d6c-9542-d914e95b3df1"),
            client_id=os.getenv("DATAVERSE_CLIENT_ID",
                                "72cf461b-f9d2-4c88-a009-52abef2db39c"),
            client_secret=os.getenv(
                "DATAVERSE_CLIENT_SECRET", "fme8Q~I5JVCfGoxFktYoKonrVsH4vYfmxvPAZa7J")
        )
    except Exception as e:
        # We don't want to stop the whole app if HR DB is down,
        # the agent just won't have personalized tone.
        st.sidebar.warning(f"HR Database connection failed: {e}")

    user_email = "Khaled.Arafa@Andalusiagroup.net"
    return OPDHealthcareAgent(data_model, llm, hr_db=hr_db, user_email=user_email)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi. Ask me about OPD revenue, BU performance, doctor rankings, no-shows, "
                    "retention, leakage, root causes, dashboards, or action plans."
                ),
            }
        ]
    if "last_reply" not in st.session_state:
        st.session_state.last_reply = None


initialize_state()

st.title(APP_TITLE)
st.caption(
    "Conversational analytics over the fixed OPD dataset and KPI knowledge base."
)

try:
    agent = load_agent()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. What would you like to analyze?",
            }
        ]
        if "escalation_status" in st.session_state:
            del st.session_state.escalation_status
        if "kpi_request_status" in st.session_state:
            del st.session_state.kpi_request_status
        if "entity_request_status" in st.session_state:
            del st.session_state.entity_request_status
        st.session_state.last_reply = None
        st.rerun()

    # Escalation UI Section in Sidebar
    if st.session_state.get("last_reply") and st.session_state.last_reply.escalation_payload:
        reply = st.session_state.last_reply
        payload = reply.escalation_payload

        st.divider()
        st.markdown("⚠️ **Escalation Identified**")
        st.caption(f"Target: {payload['escalation_path']}")
        st.write("Trigger notification to management?")

        c1, c2 = st.columns(2)
        if c1.button("Yes", type="primary", key="side_esc_yes", use_container_width=True):
            st.toast("🔄 Triggering escalation...", icon="⏳")

            if not POWER_AUTOMATE_URL:
                st.session_state.escalation_status = "❌ Power Automate URL not configured."
                st.rerun()

            # Extract KPI name for task title
            original_task_title = payload.get("task_title", "")
            kpi_match = re.search(
                r"Performance Escalation: (.*) in", original_task_title)
            kpi_name = kpi_match.group(
                1).strip() if kpi_match else "Unknown KPI"
            new_task_title = f"{kpi_name} - Underperforming / Needs Attention"

            try:
                response = trigger_flow(
                    flow_url=POWER_AUTOMATE_URL,
                    task_title=new_task_title,
                    task_description=payload.get("task_description", ""),
                    assignee_email=payload.get("assignee_email", ""),
                    manager_email=payload.get("manager_email", ""),
                    raised_by_email=payload.get("raised_by_email", ""),
                    escalate_to_email=payload.get("escalate_to_email", ""),
                    due_date=payload.get("due_date", ""),
                    start_date=payload.get("start_date", ""),
                    task_source=payload.get("task_source", ""),
                    specialty=payload.get("specialty", ""),
                    bu_name=payload.get("bu_name", ""),
                    is_escalate=payload.get("is_escalate", True)
                )

                if response and response.status_code in (200, 202):
                    st.session_state.escalation_status = "✅ Escalation triggered successfully."
                    st.session_state.last_reply.escalation_payload = None
                else:
                    status = response.status_code if response else 'No response'
                    st.session_state.escalation_status = f"❌ Failed (Status: {status})."
            except Exception as e:
                st.session_state.escalation_status = f"❌ Connection error: {e}"
            st.rerun()

        if c2.button("No", key="side_esc_no", use_container_width=True):
            st.session_state.last_reply.escalation_payload = None
            st.session_state.escalation_status = None
            st.rerun()

    elif st.session_state.get("escalation_status"):
        st.divider()
        if "✅" in st.session_state.escalation_status:
            st.success(st.session_state.escalation_status)
        else:
            st.error(st.session_state.escalation_status)

        if st.button("Clear Status", key="clear_esc_status"):
            st.session_state.escalation_status = None
            st.rerun()

    # KPI Request UI Section in Sidebar
    if st.session_state.get("last_reply") and st.session_state.last_reply.missing_kpi_request_payload:
        reply = st.session_state.last_reply
        kpi_payload = reply.missing_kpi_request_payload

        st.divider()
        st.markdown("🔍 **KPI Not Found**")
        st.caption(
            f"Request to add: {kpi_payload.get('KPI_Title', 'New KPI')}")
        st.write("Submit request to the BI team?")

        ck1, ck2 = st.columns(2)
        if ck1.button("Request", type="primary", key="side_kpi_yes", use_container_width=True):
            st.toast("🔄 Sending KPI request...", icon="⏳")

            if not KPI_REQUEST_FLOW_URL:
                st.session_state.kpi_request_status = "❌ KPI Flow URL not configured."
                st.rerun()

            try:
                response = trigger_kpi_request_flow(
                    flow_url=KPI_REQUEST_FLOW_URL,
                    payload=kpi_payload
                )

                if response and response.status_code in (200, 202):
                    st.session_state.kpi_request_status = "✅ KPI request sent successfully."
                    st.session_state.last_reply.missing_kpi_request_payload = None
                else:
                    status = response.status_code if response else 'No response'
                    st.session_state.kpi_request_status = f"❌ Request failed (Status: {status})."
            except Exception as e:
                st.session_state.kpi_request_status = f"❌ Connection error: {e}"
            st.rerun()

        if ck2.button("Dismiss", key="side_kpi_no", use_container_width=True):
            st.session_state.last_reply.missing_kpi_request_payload = None
            st.session_state.kpi_request_status = None
            st.rerun()

    elif st.session_state.get("kpi_request_status"):
        st.divider()
        if "✅" in st.session_state.kpi_request_status:
            st.success(st.session_state.kpi_request_status)
        else:
            st.error(st.session_state.kpi_request_status)

        if st.button("Clear KPI Status", key="clear_kpi_status"):
            st.session_state.kpi_request_status = None
            st.rerun()

    # Entity Request UI Section in Sidebar
    if st.session_state.get("last_reply") and st.session_state.last_reply.missing_entity_payload:
        reply = st.session_state.last_reply
        ent_payload = reply.missing_entity_payload

        st.divider()
        st.markdown("🏢 **Missed Data Entity**")
        st.caption(f"Detected: {ent_payload.get('Entity_Name')}")

        data_type = st.selectbox(
            "Entity Category:",
            ["BU", "Doctor", "Clinic", "Department", "Specialty"],
            key="entity_type_dropdown"
        )

        if st.button("Submit Data Request", type="primary", use_container_width=True):
            ent_payload["Data_Type"] = data_type
            try:
                response = trigger_entity_request_flow(
                    ENTITY_REQUEST_FLOW_URL, ent_payload)
                if response and response.status_code in (200, 202):
                    st.session_state.entity_request_status = f"✅ Request for {data_type} submitted."
                    st.session_state.last_reply.missing_entity_payload = None
                else:
                    st.session_state.entity_request_status = "❌ Flow submission failed."
            except Exception as e:
                st.session_state.entity_request_status = f"❌ Error: {e}"
            st.rerun()

    elif st.session_state.get("entity_request_status"):
        st.divider()
        if "✅" in st.session_state.entity_request_status:
            st.success(st.session_state.entity_request_status)
        else:
            st.error(st.session_state.entity_request_status)

        if st.button("Clear Entity Status", key="clear_ent_status"):
            st.session_state.entity_request_status = None
            st.rerun()

    st.subheader("Scope")
    bu_filter = st.selectbox("BU", ["All"] + agent.model.bus, index=0)
    doctor_filter = st.selectbox(
        "Doctor", ["All"] + agent.model.doctors, index=0)
    year_filter = st.selectbox(
        "Year", ["All"] + [str(y) for y in agent.model.years], index=0)
    apply_scope = st.checkbox("Apply scope to question", value=True)

    st.subheader("Examples")
    st.markdown(
        """
- Which BU performed best in 2024?
- Who is the doctor with the highest total revenue?
- Why is Total Revenue underperforming in SMH 2024?
- Compare ASH vs SMH vs HJH on financial KPIs.
- Show me the trend chart for No-Show % across all BUs.
- Forecast Total Revenue for the next 3 months for ASH.
"""
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask about OPD KPIs, root causes, doctors, BUs, charts, or action plans"
)

if question:
    if apply_scope and (bu_filter != "All" or doctor_filter != "All" or year_filter != "All"):
        scope_parts = []
        if bu_filter != "All":
            scope_parts.append(f"BU={bu_filter}")
        if doctor_filter != "All":
            scope_parts.append(f"Doctor={doctor_filter}")
        if year_filter != "All":
            scope_parts.append(f"Year={year_filter}")
        question = f"{question}\n\nScope: {', '.join(scope_parts)}"

    # Clear old escalation status when a new question is asked
    st.session_state.escalation_status = None
    st.session_state.kpi_request_status = None
    st.session_state.entity_request_status = None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing live OPD data and KPI knowledge..."):
            try:
                reply = agent.ask(question)
            except Exception as exc:
                st.error(f"Agent failed while answering: {exc}")
                st.stop()

            st.session_state.last_reply = reply
            st.markdown(reply.answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.last_reply.answer}
    )
    st.rerun()

# 2. Render charts and evidence for the last reply in the main area
if st.session_state.get("last_reply"):
    reply = st.session_state.last_reply
    if reply.charts:
        for fig in reply.charts:
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Evidence used"):
        st.code(reply.context)

    for name, table in reply.tables.items():
        if not table.empty:
            with st.expander(name.replace("_", " ").title()):
                st.dataframe(table, use_container_width=True)
