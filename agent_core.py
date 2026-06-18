from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import json
import difflib
import requests
from typing import Any, Optional

import numpy as np
from openai import OpenAI
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from hr_user_db import EmployeeHRDB

from conversation_memory import ConversationManager
# Import the RAG module
from policy_rag import PolicyRAG

from dotenv import load_dotenv
load_dotenv(".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

SYSTEM_PROMPT = """You are a Healthcare Business Intelligence Agent for Andalusia Medical Group OPD operations.
Your job is to answer using the live operational data context and KPI knowledge context supplied by tools.

Rules:
- Answer only what was asked.
- Use the numbers from LIVE DATA CONTEXT. Never invent figures.
- If asked for root cause, list causes with data evidence.
- If asked for recommendations, generate actions from evidence. Do not copy generic templates.
- If asked for comparison, compare with exact side-by-side numbers.
- If asked for a strategic plan, give a practical 30/60/90-day roadmap.
- Use real BU and doctor names from the data.
- Be concise, specific, executive-friendly, and operationally actionable.
- Do not reveal chain-of-thought or hidden reasoning. Provide only the final answer.
- **CRITICAL**: If a KPI is below the target or threshold defined in the knowledge base or playbook, explicitly state this (e.g., 'under target' or 'below the 80% threshold'). Additionally, if the knowledge base or playbook specifies an 'Action Owner' or 'Escalation Level' for this scenario, explicitly name that role in your answer (e.g., 'escalate to Medical Manager', 'notify OPD Manager'). This is required to trigger management escalation workflows.
- **MISSING DATA**: If the user asks for a Business Unit or Doctor that is not present in the dataset, or a KPI that is absent from both the Knowledge Base AND the Policy Context, clearly state this to the user AND trigger the corresponding tool (request_missing_kpi or request_missing_entity) to help the user resolve the gap. If the information is available in the Policy Context (e.g., operational rules, hours, or bonus structures), provide the answer directly using the policy details.
"""

SUM_KPIS = {
    "Total Revenue",
    "Target Revenue",
    "Credit Revenue",
    "Cash Revenue",
    "Total Leakage Revenue Losses",
    "Total Losses Revenue_Cancellation_Modification",
    "No. Cases",
    "Target No. cases",
    "No. Services",
    "No. Booking",
    "No. Planned booking Slots",
    "No. follow-up visits",
    "No. Missed Opportunity",
    "No. Cancelled Clinics",
}

AVG_KPIS = {
    "Doctor PMS %",
    "Charge per case",
    "Service Leakage %",
    "Cross Referral %",
    "Patient Retention %",
    "Patient Acquisition %",
    "Actual COE Compliance %",
    "Digital Actual CR%",
    "Digital Target CR%",
    "No-Show %",
}

KPI_DRIVER_GRAPH = {
    "Total Revenue": {
        "drivers": [
            "No. Cases",
            "Charge per case",
            "Total Leakage Revenue Losses",
            "No. Cancelled Clinics",
            "No-Show %",
        ],
        "direction": {
            "No. Cases": "low",
            "Charge per case": "low",
            "Total Leakage Revenue Losses": "high",
            "No. Cancelled Clinics": "high",
            "No-Show %": "high",
        },
    },
    "No. Cases": {
        "drivers": [
            "No. Booking",
            "No-Show %",
            "Patient Retention %",
            "Patient Acquisition %",
            "Cross Referral %",
            "Digital Actual CR%",
        ],
        "direction": {
            "No. Booking": "low",
            "No-Show %": "high",
            "Patient Retention %": "low",
            "Patient Acquisition %": "low",
            "Cross Referral %": "low",
            "Digital Actual CR%": "low",
        },
    },
    "Patient Retention %": {
        "drivers": ["No. follow-up visits", "Doctor PMS %", "No-Show %"],
        "direction": {
            "No. follow-up visits": "low",
            "Doctor PMS %": "low",
            "No-Show %": "high",
        },
    },
    "Service Leakage %": {
        "drivers": ["No. Missed Opportunity", "Workflow Compliance (inferred)"],
        "direction": {"No. Missed Opportunity": "high"},
    },
    "Doctor PMS %": {
        "drivers": ["Patient Retention %", "Patient Acquisition %", "Cross Referral %"],
        "direction": {
            "Patient Retention %": "low",
            "Patient Acquisition %": "low",
            "Cross Referral %": "low",
        },
    },
    "No-Show %": {
        "drivers": ["No. Booking", "No. Planned booking Slots", "No. Cancelled Clinics"],
        "direction": {
            "No. Booking": "low",
            "No. Planned booking Slots": "low",
            "No. Cancelled Clinics": "high",
        },
    },
    "Charge per case": {
        "drivers": ["Service Mix (inferred)", "Insurance Mix (inferred)"],
        "direction": {"Service Mix": "low", "Insurance Mix": "high"},
    },
    "Cross Referral %": {
        "drivers": ["Doctor PMS %", "Actual COE Compliance %"],
        "direction": {"Doctor PMS %": "low", "Actual COE Compliance %": "low"},
    },
}

KPI_ALIASES: dict[str, tuple[str, ...]] = {
    "Total Revenue": ("total revenue", "revenue", "overall revenue", "revenue gap"),
    "Target Revenue": ("target revenue", "revenue target", "goal revenue"),
    "Credit Revenue": ("credit revenue", "insured revenue", "credit sales"),
    "Cash Revenue": ("cash revenue", "self pay revenue", "cash sales"),
    "Total Leakage Revenue Losses": (
        "revenue leakage",
        "leakage losses",
        "leakage loss",
        "revenue loss from leakage",
    ),
    "Total Losses Revenue_Cancellation_Modification": (
        "cancellation modification losses",
        "losses revenue cancellation modification",
        "cancellation losses",
        "modification losses",
    ),
    "No. Cases": ("no. cases", "cases", "case volume", "patient cases", "case gap"),
    "Target No. cases": ("target no cases", "target cases", "case target"),
    "No. Services": ("no. services", "services", "service volume", "service count"),
    "No. Booking": ("no. booking", "bookings", "booking count", "appointments"),
    "No. Planned booking Slots": (
        "no. planned booking slots",
        "planned booking slots",
        "planned slots",
        "available slots",
    ),
    "No. follow-up visits": (
        "no. follow up visits",
        "follow up visits",
        "follow-up visits",
        "follow ups",
    ),
    "No. Missed Opportunity": (
        "no. missed opportunity",
        "no missed opportunity",
        "missed opportunity",
        "missed opportunities",
        "missed opp",
    ),
    "No. Cancelled Clinics": (
        "no. cancelled clinics",
        "cancelled clinics",
        "cancelled clinic count",
        "clinic cancellations",
    ),
    "Doctor PMS %": ("doctor pms", "doctor pms %", "pms", "performance management score"),
    "Charge per case": ("charge per case", "avg charge per case", "price per case"),
    "Service Leakage %": (
        "service leakage",
        "leakage rate",
        "leakage percent",
        "leakage percentage",
    ),
    "Cross Referral %": ("cross referral", "referral rate", "cross referral %"),
    "Patient Retention %": (
        "patient retention",
        "retention rate",
        "return patient rate",
        "patient retention %",
    ),
    "Patient Acquisition %": (
        "patient acquisition",
        "acquisition rate",
        "new patient rate",
        "patient acquisition %",
    ),
    "Actual COE Compliance %": (
        "actual coe compliance",
        "coe compliance",
        "compliance rate",
    ),
    "Digital Actual CR%": (
        "digital actual cr",
        "actual cr",
        "digital conversion rate",
    ),
    "Digital Target CR%": (
        "digital target cr",
        "target cr",
        "digital target conversion rate",
    ),
    "No-Show %": ("no show", "no-show", "noshow", "no show percent", "no show rate"),
}

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "request_missing_kpi",
            "description": "Call this when the user asks for a metric or KPI that is not present in the data context or knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kpi_name": {"type": "string", "description": "The name of the missing KPI."}
                },
                "required": ["kpi_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_missing_entity",
            "description": "Call this when the user asks for data regarding a specific Doctor or Business Unit that is missing from the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The name of the missing BU or Doctor."}
                },
                "required": ["entity_name"]
            }
        }
    }
]

OPD_KPI_COLUMNS = [
    "Total Revenue",
    "Target Revenue",
    "Credit Revenue",
    "Cash Revenue",
    "Total Leakage Revenue Losses",
    "Total Losses Revenue_Cancellation_Modification",
    "Doctor PMS %",
    "No. Cases",
    "Target No. cases",
    "No. Services",
    "Charge per case",
    "No. Booking",
    "No. Planned booking Slots",
    "No. follow-up visits",
    "Service Leakage %",
    "Cross Referral %",
    "Patient Retention %",
    "Patient Acquisition %",
    "Actual COE Compliance %",
    "Digital Actual CR%",
    "Digital Target CR%",
    "No. Missed Opportunity",
    "No. Cancelled Clinics",
    "No-Show %",
]

LOWER_IS_BETTER_KPIS = {
    "No-Show %",
    "Service Leakage %",
    "Total Leakage Revenue Losses",
    "No. Cancelled Clinics",
    "No. Missed Opportunity",
    "Total Losses Revenue_Cancellation_Modification",
}

# ========== KPI → Policy map (user provided) ==========
kpi_policies_map = {
    "Target Revenue": [],
    "Target No. cases": [],
    "Total Revenue": ["Preoperative ECG & Cardiology Consultation Pathway in OPD"],
    "Credit Revenue": [],
    "Cash Revenue": [],
    "Total Leakage Revenue Losses": [],
    "Doctor PMS %": [
        "Doctor Time Management Policy",
        "OPD Doctor Attendance policy",
        "الاتصال بأطباء العيادات الخارجية",
        "OPD Delay policy",
        "سياسة تنظيم حضور الاطباء بالعيادات الخارجية",
    ],
    "No. Cases": [
        "OPD schedule management",
        "OPD Doctor Attendance policy",
        "OPD.03 Outpatient Assessment",
        "OPD daily Operation Service policy",
        "OPD.02 Assessment of patient for OPD surgical",
        "سياسة تنظيم حضور الاطباء بالعيادات الخارجية",
    ],
    "No. Services": [
        "Preoperative ECG & Cardiology Consultation Pathway in OPD",
        "OPD Procedure scope of practice",
        "OPD.12 ASHH Pediatric Procedure Room",
        "Dental Policy",
        "Dental procedure policy",
    ],
    "Charge per case": [
        "Preoperative ECG & Cardiology Consultation Pathway in OPD",
        "OPD Procedure scope of practice",
        "OPD.03 Outpatient Assessment",
        "OPD.12 ASHH Pediatric Procedure Room",
        "OPD.02 Assessment of patient for OPD surgical",
        "Dental Policy",
        "Dental procedure policy",
    ],
    "No. Booking": [
        "OPD schedule management",
        "OPD clinic Reservation",
        "سياسات وإجراءات الحجز داخل العيادات",
    ],
    "No. Planned booking Slots": [
        "OPD schedule management",
        "OPD clinic Reservation",
        "سياسات وإجراءات الحجز داخل العيادات",
    ],
    "No. follow-up visits": ["DAMA Retention Process (Signed)"],
    "Service Leakage %": [
        "OPD Cancellations and Modifications policy",
        "OPD Procedure scope of practice",
        "تحويل المريض من العيادات إلى الأقسام الداخلية",
        "OPD daily Operation Service policy",
        "OPD.02 Assessment of patient for OPD surgical",
        "Documentation in OPD",
    ],
    "Cross Referral %": [
        "Inflammatory Bowel Disease Center of Excellence Policy",
        "Headache & Dizziness Center of Excellence Clinical Policy",
        "Bronchial Asthma Center of Excellence Policy",
        "Diabetes Mellitus Center of Excellence Policy",
        "OPD Cross referral policy",
        "تحويل المريض من العيادات إلى الأقسام الداخلية",
    ],
    "Patient Retention %": [
        "Inflammatory Bowel Disease Center of Excellence Policy",
        "Headache & Dizziness Center of Excellence Clinical Policy",
        "Bronchial Asthma Center of Excellence Policy",
        "Diabetes Mellitus Center of Excellence Policy",
        "Patient Waiting time policy “Queue System”",
        "DAMA Retention Process (Signed)",
        "OPD Cross referral policy",
        "OPD.03 Outpatient Assessment",
        "OPD Delay policy",
        "OPD daily Operation Service policy",
        "Documentation in OPD",
    ],
    "Patient Acquisition %": [
        "Patient Waiting time policy “Queue System”",
        "Dental Policy",
    ],
    "Actual COE Compliance %": [
        "Inflammatory Bowel Disease Center of Excellence Policy",
        "Headache & Dizziness Center of Excellence Clinical Policy",
        "Bronchial Asthma Center of Excellence Policy",
        "Diabetes Mellitus Center of Excellence Policy",
        "OPD Cross referral policy",
        "Documentation in OPD",
    ],
    "Digital Actual CR%": [],
    "Digital Target CR%": [],
    "No. Missed Opportunity": [],
    "No. Cancelled Clinics": ["OPD Cancellations and Modifications policy"],
    "Total Losses Revenue_Cancellation_Modification": ["OPD Cancellations and Modifications policy"],
    "No-Show %": [
        "Doctor Time Management Policy",
        "Patient Waiting time policy “Queue System”",
        "OPD Doctor Attendance policy",
        "الاتصال بأطباء العيادات الخارجية",
        "OPD Delay policy",
        "سياسة تنظيم حضور الاطباء بالعيادات الخارجية",
    ],
}

bu_scope = {
    "AES": "ALL",
    "AHBS": "ALL",
    "AHJ": "HJH",
    "AHQ": "ALL",
    "AMH": "AMH",
    "ARC": "SMH",
    "ASH": "ASH",
    "Alex": ["ASH", "SMH"],
    "CHQ": "HJH",
    "EGY": ["ASH", "SMH"],
    "ERS": "HJH",
    "GEO": "ALL",
    "SMH": "SMH",
    "Venture": ["ASH", "SMH"],
    "org2f45e702": "ALL"
}

# ========== Load policy summaries ==========
POLICY_SUMMARIES_PATH = Path(__file__).parent / "policy_summaries.json"
try:
    with open(POLICY_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        POLICY_SUMMARIES = json.load(f)
except FileNotFoundError:
    POLICY_SUMMARIES = {}
    print("Warning: policy_summaries.json not found. Policy injection disabled.")

# ========== Singleton RAG instance ==========
_RAG_INSTANCE = None


def _get_rag() -> PolicyRAG | None:
    """
    Query-only RAG loader. Reads from the pre-built ChromaDB on disk.
    Policy documents are already embedded; the query embedder is loaded once.
    """
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        if not POLICY_SUMMARIES:
            print("[PolicyRAG] Warning: POLICY_SUMMARIES is empty. Policy injection disabled.")
            return None
        try:
            backend = "sentence_transformer" if not os.getenv(
                "EMBEDDING_SERVER_URL") else "chroma_default"
            _RAG_INSTANCE = PolicyRAG(
                json_path=POLICY_SUMMARIES_PATH,
                persist_dir=Path(__file__).parent / "chroma_db",
                similarity_threshold=0.30,
                query_backend=backend,
                chroma_host=os.getenv("CHROMA_HOST"),
                embedding_server_url=os.getenv("EMBEDDING_SERVER_URL"),
            )
            print("RAG Instance loaded successfully.")
        except Exception as e:
            print(f"[PolicyRAG] Initialization failed: {e}")
            _RAG_INSTANCE = None
    return _RAG_INSTANCE


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _resolve_kpi_from_text(text: str, candidate_kpis: list[str] | None = None) -> str | None:
    normalized_text = _normalize_text(text)
    candidates = candidate_kpis or []
    words = normalized_text.split()

    # 1. Direct Alias matching (exact or substring)
    alias_pairs: list[tuple[str, str]] = []
    for canonical, aliases in KPI_ALIASES.items():
        for alias in aliases:
            alias_pairs.append((_normalize_text(alias), canonical))

    alias_pairs.sort(key=lambda item: len(item[0]), reverse=True)

    for alias, canonical in alias_pairs:
        if alias in normalized_text:
            if not candidates or canonical in candidates:
                return canonical

    # 2. Direct Candidate matching (exact or substring)
    if candidates:
        for kpi in sorted(candidates, key=len, reverse=True):
            if _normalize_text(kpi) in normalized_text:
                return kpi

    # 3. Fuzzy Fallback: Handle misspelled KPI names by checking similarities
    all_targets = {}
    # 3. Fuzzy Fallback: Improved phrase-based matching
    # Create a flat map of all normalized aliases to canonical names
    target_map = {}
    for canonical, aliases in KPI_ALIASES.items():
        for a in aliases:
            norm_a = _normalize_text(a)
            all_targets[norm_a] = canonical
            # Break down multi-word aliases to catch typos in individual words (e.g. "acuisision" -> "acquisition")
            for part in norm_a.split():
                if len(part) > 3:
                    all_targets[part] = canonical
            target_map[_normalize_text(a)] = canonical
    for c in candidates:
        all_targets[_normalize_text(c)] = c
        target_map[_normalize_text(c)] = c

    for word in words:
        if len(word) < 4:
            continue
        matches = difflib.get_close_matches(word, list(all_targets.keys()), n=1, cutoff=0.75)
        if matches:
            return all_targets[matches[0]]
    query_words = normalized_text.split()
    target_phrases = list(target_map.keys())
    
    # Prioritize multi-word segments to avoid collisions on common words like "Patient" or "Revenue"
    for n in range(3, 0, -1):
        for i in range(len(query_words) - n + 1):
            segment = " ".join(query_words[i:i+n])
            if len(segment) > 4:
                matches = difflib.get_close_matches(segment, target_phrases, n=1, cutoff=0.75)
                if matches:
                    return target_map[matches[0]]

    # Final fallback for single word typos if phrase matching failed
    for word in query_words:
        if len(word) > 5:
            matches = difflib.get_close_matches(word, target_phrases, n=1, cutoff=0.8)
            if matches:
                return target_map[matches[0]]

    return None


def _resolve_metric_column(frame: pd.DataFrame, kpi: str | None) -> str | None:
    if frame.empty or not kpi:
        return None
    if kpi in frame.columns:
        return kpi

    normalized_to_col = {_normalize_text(col): col for col in frame.columns}
    candidates: list[str] = [kpi]
    if kpi in KPI_ALIASES:
        candidates.extend(KPI_ALIASES[kpi])
    for canonical, aliases in KPI_ALIASES.items():
        if _normalize_text(kpi) == _normalize_text(canonical) or any(
            _normalize_text(kpi) == _normalize_text(alias) for alias in aliases
        ):
            candidates.append(canonical)
            candidates.extend(aliases)

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in candidates:
        key = _normalize_text(candidate)
        if key and key not in seen:
            seen.add(key)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate in normalized_to_col:
            return normalized_to_col[normalized_candidate]

    normalized_kpi = _normalize_text(kpi)
    for normalized_col, original_col in normalized_to_col.items():
        if normalized_kpi and normalized_kpi in normalized_col:
            return original_col

    return None


def _resolve_rank_sorting(kpi: str, rank_preference: str | None) -> bool:
    if rank_preference == "highest":
        return False
    if rank_preference == "lowest":
        return True
    if rank_preference == "best":
        return kpi in LOWER_IS_BETTER_KPIS
    return kpi in LOWER_IS_BETTER_KPIS


def _get_user_based_system_prompt(base_prompt: str, user_profile: dict | None, memory_context: str = "") -> str:
    """
    Adjusts the system prompt based on the user's identity details fetched from Dataverse.
    Now includes a memory context block for daily summaries and KPIs. """
    if not user_profile:
        return base_prompt

    first_name = user_profile.get('hr_firstname', 'User')
    user_name = user_profile.get('hr_fullname', 'User')
    org_title = user_profile.get(
        'org_display_name', user_profile.get('hr_adtitle', 'Staff'))
    dept = user_profile.get('hr_department', 'General')
    bu = user_profile.get('bu', 'All')
    job_title_lower = (user_profile.get('hr_jobtitle') or '').lower()
    is_doctor = user_profile.get('doctor_name') is not None

    # Resolve effective scope using bu_scope map
    effective_scope = bu_scope.get(bu, bu)
    print(
        f"User '{user_name}' has BU '{bu}' with effective scope '{effective_scope}'.")
    all_access_keywords = ["ALL", "CORPORATE",
                           "GROUP", "ANDALUSIA GROUP", "AH", "AHQ"]
    is_all_access = (effective_scope == "ALL") or (
        str(bu).upper() in all_access_keywords)

    identity_block = (
        f"USER IDENTITY CONTEXT:\n"
        f"- Greeting Name: {first_name}\n"
        f"- Full Name: {user_name}\n"
        f"- Job Title: {org_title}\n"
        f"- Department: {dept}\n"
        f"- BU: {bu} (Effective Scope: {effective_scope})\n"
        f"- Role Type: {'Clinician' if is_doctor else 'Non-Clinician Staff'}"
    )

    if not is_doctor:
        identity_block += f"\nCRITICAL: The user is a Non-Clinician. STRICTLY FORBIDDEN: Do not mention, compare, or refer to any doctor in the dataset who shares the same name as the user. Identify the user solely by their HR profile. Never acknowledge a namesake doctor from the live data in your response to this user."

    if not is_all_access:
        scope_str = ", ".join(effective_scope) if isinstance(
            effective_scope, list) else effective_scope
        identity_block += f"\nCRITICAL: The user is restricted to Business Unit scope '{scope_str}'. STRICTLY FORBIDDEN: Do not mention, compare, or display any data or numbers related to other Business Units. All analysis must be exclusively for {scope_str}."

    print(identity_block)

    memory_block = ""
    if memory_context:
        memory_block = f"\nDAILY CONVERSATION MEMORY (Prior interactions today):\n{memory_context}\n"
        print(
            "[SystemPrompt] Successfully appended daily memory context to the system prompt.")

    tone_adjustment = f"Instruction: Use the user's Greeting Name ({first_name}) to address them directly for engagement. If the user is Non-Clinician, strictly ignore data regarding doctors with the same name.\n"
    if any(k in job_title_lower for k in ["manager", "director", "head", "lead"]):
        tone_adjustment += (
            "Instruction: Provide professional, insightful, and comprehensive executive-friendly answers. Focus on high-level summaries and strategic implications. Address the user with a balanced tone of urgency and professionalism when data is underperforming."
        )
    elif any(k in job_title_lower for k in ["engineer", "analyst", "specialist", "technician", "developer"]):
        tone_adjustment += (
            "Instruction: Provide detailed, data-driven answers including specific figures, deep-dive evidence, and technical logic."
        )

    return f"{base_prompt}\n\n{identity_block}\n{memory_block}\n\n{tone_adjustment}"


def trigger_flow(
    flow_url: str,
    task_title: str,
    task_description: str,
    assignee_email: str,
    manager_email: str,
    raised_by_email: str,
    escalate_to_email: str,
    due_date: str,
    start_date: str,
    task_source: str,
    specialty: str,
    bu_name: str,
    is_escalate: bool
) -> Optional[requests.Response]:
    """
    Trigger a Power Automate flow (HTTP trigger) with the specified task data.
    """
    body = {
        "task_title": task_title, "task_description": task_description, "assignee_email": assignee_email,
        "manager_email": manager_email, "raised_by_email": raised_by_email, "escalate_to_email": escalate_to_email,
        "due_date": due_date, "start_date": start_date, "task_source": task_source,
        "specialty": specialty, "bu_name": bu_name, "is_escalate": is_escalate
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            flow_url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"[AgentCore] Failed to trigger flow: {e}")
        return None


def trigger_kpi_request_flow(
    flow_url: str,
    payload: dict[str, Any]
) -> Optional[requests.Response]:
    """
    Trigger a Power Automate flow (HTTP trigger) for a missing KPI request.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.post(
            flow_url,
            json=payload,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"[AgentCore] Failed to trigger KPI request flow: {e}")
        if e.response is not None:
            print(f"[AgentCore] Flow Error Detail: {e.response.text}")
        return None


def trigger_entity_request_flow(
    flow_url: str,
    payload: dict[str, Any]
) -> Optional[requests.Response]:
    """
    Trigger a Power Automate flow for a missing data entity (BU, Doctor, etc.).
    """
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            flow_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"[AgentCore] Failed to trigger Entity request flow: {e}")
        return None


@dataclass
class AgentConfig:
    groq_api_key: str
    groq_model: str = DEFAULT_GROQ_MODEL
    temperature: float = 0.35
    top_p: float = 0.92
    max_tokens: int = 1800


@dataclass
class AgentReply:
    answer: str
    charts: list[go.Figure] = field(default_factory=list)
    context: str = ""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    escalation_payload: dict | None = None
    missing_kpi_request_payload: dict | None = None
    missing_entity_payload: dict | None = None


class GroqLLM:
    def __init__(self, config: AgentConfig) -> None:
        if not config.groq_api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Add it in the UI or .env.")
        self.config = config
        self.client = OpenAI(base_url=GROQ_BASE_URL,
                             api_key=config.groq_api_key)

    def _estimate_tokens(self, text: str) -> int:
        """
        A rough estimation of tokens using word count.
        This is a heuristic as actual tokenization is model-dependent and complex.
        For many models, ~1 word roughly equals ~1.3 tokens, but simple word count
        gives a quick estimate for input size.
        """
        return len(text.split()) if text else 0

    def generate(self, user_prompt: str, system_prompt_override: str | None = None, tools: list | None = None, max_tokens: int | None = None) -> tuple[str, list]:
        final_system_prompt = system_prompt_override if system_prompt_override is not None else SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Estimate input token size BEFORE the API call
        estimated_system_tokens = self._estimate_tokens(final_system_prompt)
        estimated_user_tokens = self._estimate_tokens(user_prompt)
        print(f"[LLM Token Estimation] Estimated input tokens: {estimated_system_tokens + estimated_user_tokens} (System: {estimated_system_tokens}, User: {estimated_user_tokens})")

        token_limit = max_tokens if max_tokens is not None else self.config.max_tokens
        kwargs = {
            "model": self.config.groq_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": token_limit,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""

        # Log token usage statistics
        if hasattr(response, 'usage') and response.usage:
            u = response.usage
            print(
                f"[LLM Usage] Input: {u.prompt_tokens} tokens | Output: {u.completion_tokens} tokens | Total: {u.total_tokens}"
            )

        content = re.sub(r"<think>.*?</think>", "", content,
                         flags=re.DOTALL | re.IGNORECASE)
        tool_calls = msg.tool_calls or []
        return content.strip(), tool_calls


class OPDDataModel:
    def __init__(self, opd_path: str | Path, knowledge_path: str | Path, db_url: str | None = None) -> None:
        self.opd_path = Path(opd_path)
        self.knowledge_path = Path(knowledge_path)
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.df, self.doctor_map = self._load_opd(self.opd_path)
        self.kb = self._load_knowledge(self.knowledge_path)
        self.numeric_kpis = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in {"Year", "Month No"}
        ]
        self.kpi_columns = [
            column for column in OPD_KPI_COLUMNS if column in self.df.columns]
        for column in self.numeric_kpis:
            if column not in self.kpi_columns and column not in {"Year", "Month No"}:
                self.kpi_columns.append(column)
        self.bus = sorted(self.df["BU"].dropna().astype(str).unique().tolist())
        self.doctors = sorted(
            self.df["Doctor Name"].dropna().astype(str).unique().tolist()
        )
        self.years = sorted(
            self.df["Year"].dropna().astype(int).unique().tolist())

    def _load_opd(self, path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
        df = pd.read_excel(path, sheet_name="OPD_KPI_Dataset",
                           parse_dates=["Month"])
        df["Month_Label"] = df["Month"].dt.strftime("%b %Y")
        df["YearMonth"] = df["Month"].dt.to_period("M").astype(str)
        for column in OPD_KPI_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        # Logic to aggregate doctors with different naming entries (e.g. "Khaled Hassan" vs "Khaled Mohamed Hassan")
        raw_names = df["Doctor Name"].dropna().astype(str).unique().tolist()
        # Sort by length descending to prefer longer, more complete names as canonical
        sorted_raw = sorted(raw_names, key=len, reverse=True)
        mapping = {}
        processed = set()

        for name in sorted_raw:
            if name in processed:
                continue
            mapping[name] = name
            processed.add(name)
            name_norm = _normalize_text(name)
            name_tokens = set(name_norm.split())

            for other in sorted_raw:
                if other in processed:
                    continue
                other_norm = _normalize_text(other)
                other_tokens = set(other_norm.split())

                # Check for subset (tokens of short name in long name) or First+Last name match
                is_match = False
                if other_tokens.issubset(name_tokens) and len(other_tokens) >= 2:
                    is_match = True
                elif len(name_tokens) >= 2 and len(other_tokens) >= 2:
                    n_list, o_list = name_norm.split(), other_norm.split()
                    if n_list[0] == o_list[0] and n_list[-1] == o_list[-1]:
                        is_match = True

                if is_match:
                    mapping[other] = name
                    processed.add(other)

        df["Doctor Name"] = df["Doctor Name"].map(mapping)
        return df, mapping

    def _load_knowledge(self, path: Path) -> dict[str, pd.DataFrame]:
        book = pd.ExcelFile(path)
        return {sheet: pd.read_excel(book, sheet_name=sheet) for sheet in book.sheet_names}

    @property
    def kb_map(self) -> pd.DataFrame:
        return self.kb.get("adx_kpi_knowledge_map_x0009__x0009__x0009_", pd.DataFrame())

    @property
    def kb_playbook(self) -> pd.DataFrame:
        return self.kb.get("adx_kpi_investigation_playbook", pd.DataFrame())

    @property
    def kb_rel(self) -> pd.DataFrame:
        return self.kb.get("adx_kpi_relationship_map_x0009__x0009__x0009_", pd.DataFrame())

    def filtered(self, bu: str | list[str] | None = None, doctor: str | None = None, year: int | None = None) -> pd.DataFrame:
        sub = self.df.copy()
        if bu:
            if isinstance(bu, list):
                sub = sub[sub["BU"].astype(str).str.upper().isin(
                    [b.upper() for b in bu])]
            else:
                sub = sub[sub["BU"].astype(str).str.upper() == bu.upper()]
        if doctor:
            sub = sub[sub["Doctor Name"].astype(
                str).str.lower() == doctor.lower()]
        if year:
            sub = sub[sub["Year"] == int(year)]
        return sub

    def relevant_kb(self, question: str, max_rows: int = 6) -> str:
        q = question.lower()
        keyword_map = {
            "revenue": ["Total Revenue", "Revenue"],
            "service leakage": ["Service Leakage"],
            "revenue leakage": ["Total Leakage Revenue Losses"],
            "leakage loss": ["Total Leakage Revenue Losses"],
            "leakage losses": ["Total Leakage Revenue Losses"],
            "leakage": ["Service Leakage", "Leakage"],
            "no-show": ["No-Show"],
            "noshow": ["No-Show"],
            "retention": ["Patient Retention"],
            "acquisition": ["Patient Acquisition"],
            "referral": ["Cross Referral"],
            "pms": ["Doctor PMS"],
            "cancel": ["Cancelled Clinics"],
            "digital": ["Digital"],
            "cases": ["No. Cases"],
            "booking": ["Booking"],
            "charge": ["Charge per case"],
            "delay": ["Doctor PMS", "No-Show", "Patient Retention"],
            "utilization": ["Booking", "No. Planned booking Slots"],
        }
        matched = set()
        for keyword, names in keyword_map.items():
            if keyword in q:
                matched.update(names)

        rows = []
        if not self.kb_map.empty:
            for _, row in self.kb_map.iterrows():
                kpi_name = str(row.get("KPI_Name", "")).lower()
                if not matched or any(name.lower() in kpi_name for name in matched):
                    rows.append(row)

        lines = ["[KPI KNOWLEDGE CONTEXT]"]
        for row in rows[:max_rows]:
            lines.append(
                f"KPI: {row.get('KPI_Name','')} | Driver: {row.get('Primary_Driver_KPI','')} | "
                f"Secondary: {row.get('Secondary_Driver_KPI','')} | Investigation: "
                f"{row.get('Investigation_Step_1','')} -> {row.get('Investigation_Step_2','')} -> "
                f"{row.get('Investigation_Step_3','')} | Owner: {row.get('Action_Owner','')} | "
                f"Escalation: {row.get('Escalation_Level','')}"
            )

        if not self.kb_playbook.empty:
            for _, row in self.kb_playbook.iterrows():
                kpi = str(row.get("KPI", "")).lower()
                if not matched or any(name.lower() in kpi for name in matched):
                    lines.append(
                        f"PLAYBOOK: {row.get('KPI','')} | Scenario: {row.get('Scenario','')} | "
                        f"Threshold: {row.get('Threshold','')} | Severity: {row.get('Severity','')} | "
                        f"Root focus: {row.get('Root_Cause_Focus','')} | Escalation: {row.get('Escalation','')}"
                    )
        return "\n".join(lines[: max_rows + 5])


def parse_query_context(question: str, model: OPDDataModel) -> dict[str, Any]:
    q = question.lower()
    ctx = {"bu": None, "doctor": None, "year": None, "kpi": None,
           "want_chart": False, "chart_type": None, "bus": []}
    for bu in model.bus:
        if re.search(r"\b" + re.escape(bu) + r"\b", question, re.I):
            ctx["bus"].append(bu)
    if len(ctx["bus"]) == 1:
        ctx["bu"] = ctx["bus"][0]
    # Search for all name variations but resolve to the canonical name stored in the context
    for variation, canonical in model.doctor_map.items():
        if re.search(r"\b" + re.escape(variation) + r"\b", question, re.I):
            ctx["doctor"] = canonical
            break
    for year in model.years:
        if str(year) in question:
            ctx["year"] = int(year)
            break

    # Default to current or latest year for status/general queries if none specified
    status_keywords = ["analysis", "analyze", "performance", "tracking", "summary", "overview",
                       "current", "latest", "situation", "status", "today", "now", "doing", "trend", "chart"]
    if ctx["year"] is None:
        current_year = 2026  # Context-provided current year
        if current_year in model.years:
            ctx["year"] = current_year
        elif model.years:
            ctx["year"] = model.years[-1]

    ctx["kpi"] = _resolve_kpi_from_text(question, model.numeric_kpis)
    if ctx["kpi"] is None:
        ctx["kpi"] = detect_investigation_kpi(question)
    if any(word in q for word in ["dashboard", "chart", "graph", "plot", "visual", "trend", "heatmap", "scorecard", "show", "analyze", "ranking", "forecast", "predict", "projection"]):
        ctx["want_chart"] = True
        if "heatmap" in q:
            ctx["chart_type"] = "heatmap"
        elif "scorecard" in q or "doctor" in q:
            ctx["chart_type"] = "scorecard"
        elif "trend" in q:
            ctx["chart_type"] = "trend"
        elif any(w in q for w in ["forecast", "predict", "projection"]):
            ctx["chart_type"] = "forecast"
        else:
            ctx["chart_type"] = "dashboard"
    if any(w in q for w in ["highest", "max", "maximum", "most", "worst", "top"]):
        ctx["rank_preference"] = "highest"
    elif any(w in q for w in ["lowest", "min", "minimum", "least"]):
        ctx["rank_preference"] = "lowest"
    elif "best" in q:
        ctx["rank_preference"] = "best"
    else:
        ctx["rank_preference"] = None
    return ctx


def is_investigation_query(question: str) -> bool:
    triggers = [
        "root cause",
        "why is",
        "underperforming",
        "investigate",
        "gap",
        "driver",
        "what is causing",
        "deep dive",
        "analyze",
    ]
    q = question.lower()
    return any(t in q for t in triggers)


def is_policy_query(question: str) -> bool:
    """Detect if the user is asking about a policy (its rules, definitions, consequences)."""
    q = question.lower()
    policy_triggers = [
        "what is the policy", "what's the policy", "tell me about the policy",
        "policy of", "policy for", "policy regarding", "policy on",
        "cancellation policy", "delay policy", "documentation policy",
        "attendance policy", "time management policy", "cross referral policy",
        "procedure policy", "dental policy", "coe policy", "coverage policy",
        "leave policy", "modification policy", "reservation policy",
        "waiting time policy", "schedule management policy", "scope of practice",
        "policy summary", "explain the policy", "describe the policy",
        "bonus scheme", "bonus for", "incentive", "deduction", "penalty",
        "threshold", "compliance percentage", "what is the bonus",
        "how much bonus", "what happens if", "consequences of",
        "penalty for", "allowed cancellations", "maximum planned cancellations",
        "delay threshold", "how many cancellations", "how many planned",
        "who owns", "responsible for", "escalation", "red flag",
        "salary deduction", "bonus holding", "annual leave deduction",
        "deduction for", "trigger", "automated email", "green flag", "late", "lateness",
        "unplanned cancellation", "critical cancellation", "planned cancellation",
        "modification rule", "travel buffer", "minimum buffer", "working hours",
        "weekly working hours", "back-to-back", "travel time",
        "per physician", "physician hours", "hours of work",
        "prohibited medications", "pre-procedure", "physical examination", "policy",
        "documentation compliance", "compliance %", "percentage compliance"
    ]
    return any(phrase in q for phrase in policy_triggers)


def detect_investigation_kpi(question: str) -> str | None:
    q = question.lower()
    normalized_q = _normalize_text(question)
    if "digital cr" in normalized_q or "cr rate" in normalized_q or "conversion rate" in normalized_q:
        if any(token in normalized_q for token in ["target", "goal", "planned"]):
            return "Digital Target CR%"
        return "Digital Actual CR%"

    if "revenue leakage" in normalized_q or "leakage loss" in normalized_q or "leakage losses" in normalized_q:
        return "Total Leakage Revenue Losses"

    if "service leakage" in normalized_q:
        return "Service Leakage %"

    if "missed opportunit" in normalized_q:
        return "No. Missed Opportunity"

    resolved = _resolve_kpi_from_text(question)
    if resolved:
        return resolved

    fallback_mapping = {
        "revenue": "Total Revenue",
        "leakage": "Service Leakage %",
        "retention": "Patient Retention %",
        "acquisition": "Patient Acquisition %",
        "referral": "Cross Referral %",
        "pms": "Doctor PMS %",
        "booking": "No. Booking",
        "charge": "Charge per case",
        "cancel": "No. Cancelled Clinics",
        "case": "No. Cases",
    }
    for kw, kpi in fallback_mapping.items():
        if kw in normalized_q:
            return kpi
    return None


def _agg_value(sub: pd.DataFrame, kpi: str) -> float:
    if kpi not in sub.columns:
        return float("nan")
    if kpi in SUM_KPIS:
        return float(sub[kpi].sum())
    return float(sub[kpi].mean())


def _get_target(sub: pd.DataFrame, kpi: str) -> tuple[float, bool]:
    if kpi == "Total Revenue" and "Target Revenue" in sub.columns:
        return float(sub["Target Revenue"].sum()), True
    if kpi == "No. Cases" and "Target No. cases" in sub.columns:
        return float(sub["Target No. cases"].sum()), True
    if kpi == "Digital Actual CR%" and "Digital Target CR%" in sub.columns:
        return float(sub["Digital Target CR%"].mean()), True
    return float("nan"), False


def _severity(gap_pct: float, is_bad: bool) -> str:
    if not is_bad:
        return "🟢"
    abs_gap = abs(gap_pct)
    if abs_gap >= 30:
        return "🔴"
    if abs_gap >= 15:
        return "🟠"
    if abs_gap >= 5:
        return "🟡"
    return "🟢"


def compute_gap_analysis(model: OPDDataModel, kpi: str, ctx: dict[str, Any]) -> str:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return f"No data for {kpi} with the given filters."

    actual = _agg_value(sub, kpi)
    target, has_target = _get_target(sub, kpi)
    unit = "SAR" if ("Revenue" in kpi or "Losses" in kpi) else (
        "%" if "%" in kpi else "units")
    scope = f"BU={ctx.get('bu') or 'All'} | Doctor={ctx.get('doctor') or 'All'} | Year={ctx.get('year') or 'All'}"

    lines = [f"GAP ANALYSIS — {kpi}", f"Scope: {scope}"]
    if unit == "SAR":
        lines.append(f"  Actual : {actual:,.0f} SAR")
    elif unit == "%":
        lines.append(f"  Actual : {actual*100:.2f}%")
    else:
        lines.append(f"  Actual : {actual:,.0f}")

    if has_target:
        target_gap = actual - target
        target_pct = (target_gap / target * 100) if target != 0 else 0
        if unit == "SAR":
            lines.append(f"  Target : {target:,.0f} SAR")
            lines.append(
                f"  Gap    : {target_gap:+,.0f} SAR ({target_pct:+.1f}%)")
        elif unit == "%":
            lines.append(f"  Target : {target*100:.2f}%")
            lines.append(
                f"  Gap    : {target_gap*100:+.2f} pp ({target_pct:+.1f}%)")
        else:
            lines.append(f"  Target : {target:,.0f}")
            lines.append(f"  Gap    : {target_gap:+,.0f} ({target_pct:+.1f}%)")
    else:
        lines.append("  Target : Not defined")
    return "\n".join(lines)


def compute_driver_breakdown(model: OPDDataModel, kpi: str, ctx: dict[str, Any]) -> str:
    if kpi not in KPI_DRIVER_GRAPH:
        return f"No driver map defined for '{kpi}'."

    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return "No data for given filters."

    drivers = KPI_DRIVER_GRAPH[kpi]["drivers"]
    direction = KPI_DRIVER_GRAPH[kpi]["direction"]
    scope = f"BU={ctx.get('bu') or 'All'} | Doctor={ctx.get('doctor') or 'All'} | Year={ctx.get('year') or 'All'}"

    rows: list[dict[str, Any]] = []
    for drv in drivers:
        if drv not in sub.columns:
            continue
        actual = _agg_value(sub, drv)
        if np.isnan(actual):
            continue

        target = None
        has_target = False
        if drv == "No. Booking" and "No. Planned booking Slots" in sub.columns:
            target = float(sub["No. Planned booking Slots"].sum())
            has_target = True
        elif drv == "No. Cases" and "Target No. cases" in sub.columns:
            target = float(sub["Target No. cases"].sum())
            has_target = True

        if "%" in drv:
            act_fmt = f"{actual*100:.2f}%"
        elif drv in SUM_KPIS and ("Revenue" in drv or "Losses" in drv):
            act_fmt = f"{actual:,.0f} SAR"
        else:
            act_fmt = f"{actual:,.0f}"

        is_bad = False
        gap_pct = 0.0
        if has_target and target is not None:
            if direction.get(drv, "low") == "low":
                is_bad = actual < target
            else:
                is_bad = actual > target
            gap_pct = (actual - target) / target * 100 if target != 0 else 0

        sev = _severity(gap_pct, is_bad)
        target_fmt = "—"
        if has_target and target is not None:
            target_fmt = f"{target*100:.2f}%" if "%" in drv else f"{target:,.0f}"

        rows.append({
            "driver": drv,
            "actual": act_fmt,
            "target": target_fmt,
            "severity": sev,
            "is_bad": is_bad,
        })

    rows.sort(key=lambda r: (not r["is_bad"]))

    lines = [
        f"DRIVER VALUES — factors affecting {kpi}",
        f"Scope: {scope}",
        "",
    ]
    for idx, row in enumerate(rows, 1):
        lines.append(
            f"  {idx}. {row['severity']} {row['driver']}: Actual={row['actual']} | Target={row['target']}"
        )

    bad = [r for r in rows if r["is_bad"]]
    if bad:
        lines.append("")
        lines.append(f"⚠️  UNDERPERFORMING DRIVERS ({len(bad)}):")
        for row in bad:
            lines.append(
                f"  → {row['driver']}: {row['actual']} (target: {row['target']})"
            )

    return "\n".join(lines)


def compute_doctor_ranking(model: OPDDataModel, ctx: dict[str, Any], kpi: str) -> str:
    sub = model.filtered(ctx.get("bu"), year=ctx.get("year"))
    if sub.empty or kpi not in sub.columns:
        return f"No data for {kpi}."
    agg = "sum" if kpi in SUM_KPIS else "mean"
    grp = sub.groupby("Doctor Name", as_index=False).agg(value=(kpi, agg))

    ascending = _resolve_rank_sorting(kpi, ctx.get("rank_preference"))
    grp = grp.sort_values("value", ascending=ascending).reset_index(drop=True)
    best_val = grp.iloc[0]["value"]

    lines = [f"DOCTOR RANKING — {kpi}",
             f"Scope: BU={ctx.get('bu') or 'All'} | Year={ctx.get('year') or 'All'}"]
    for i, row in grp.iterrows():
        val = float(row["value"])
        gap = val - best_val
        if "%" in kpi:
            val_fmt = f"{val*100:.2f}%"
            gap_fmt = f"{gap*100:+.2f} pp vs #1"
        elif kpi in SUM_KPIS and ("Revenue" in kpi or "Losses" in kpi):
            val_fmt = f"{val:,.0f} SAR"
            gap_fmt = f"{gap:+,.0f} SAR vs #1"
        else:
            val_fmt = f"{val:,.1f}"
            gap_fmt = f"{gap:+.1f} vs #1"
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
        lines.append(f"  {medal} {row['Doctor Name']}: {val_fmt}  ({gap_fmt})")
    return "\n".join(lines)


def compute_bu_comparison_text(model: OPDDataModel, ctx: dict[str, Any], kpi: str) -> str:
    sub = model.filtered(year=ctx.get("year"))
    bus = ctx.get("bus") or model.bus
    lines = [f"BU COMPARISON — {kpi}",
             f"Year: {ctx.get('year') or 'All Years'}"]
    if ctx.get("bus"):
        lines.append(f"BUs: {', '.join(bus)}")

    results = []
    for bu in bus:
        bu_sub = sub[sub["BU"] == bu]
        if bu_sub.empty or kpi not in bu_sub.columns:
            continue
        val = _agg_value(bu_sub, kpi)
        target, has_target = _get_target(bu_sub, kpi)
        results.append({
            "bu": bu,
            "val": val,
            "target": target,
            "has_target": has_target
        })

    if not results:
        return f"No data for {kpi}."

    ascending = _resolve_rank_sorting(kpi, ctx.get("rank_preference"))
    results.sort(key=lambda x: x["val"], reverse=not ascending)
    best_val = results[0]["val"]

    for item in results:
        bu = item["bu"]
        val = item["val"]
        # Prioritize target gap over leader benchmarking if target exists
        if item["has_target"]:
            gap = val - item["target"]
            gap_label = "gap vs target"
        else:
            gap = val - best_val
            gap_label = "gap vs best"

        if "%" in kpi:
            lines.append(
                f"  {bu}: {val*100:.2f}%  ({gap_label}: {gap*100:+.2f} pp)"
            )
        elif "Revenue" in kpi or "Losses" in kpi:
            lines.append(
                f"  {bu}: {val:,.0f} SAR  ({gap_label}: {gap:+,.0f} SAR)")
        else:
            lines.append(f"  {bu}: {val:,.1f}  ({gap_label}: {gap:+.1f})")
    return "\n".join(lines)


def compute_trend_narrative(model: OPDDataModel, ctx: dict[str, Any], kpi: str) -> str:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty or kpi not in sub.columns:
        return f"No trend data for {kpi}."

    agg = "sum" if kpi in SUM_KPIS else "mean"
    monthly = sub.groupby(["Year", "Month No", "Month_Label"],
                          as_index=False).agg(value=(kpi, agg))
    monthly = monthly.sort_values(["Year", "Month No"]).reset_index(drop=True)
    if len(monthly) < 2:
        return f"Not enough data points for trend analysis of {kpi}."

    first_val = monthly.iloc[0]["value"]
    last_val = monthly.iloc[-1]["value"]
    change = last_val - first_val
    change_pct = (change / abs(first_val) * 100) if first_val != 0 else 0
    best_idx = monthly["value"].idxmax()
    worst_idx = monthly["value"].idxmin()

    lower_is_better = {
        "No-Show %",
        "Service Leakage %",
        "Total Leakage Revenue Losses",
        "No. Cancelled Clinics",
        "No. Missed Opportunity",
    }
    if kpi in lower_is_better:
        direction = "📉 Worsening" if change > 0 else "📈 Improving"
    else:
        direction = "📈 Improving" if change > 0 else "📉 Declining"

    if "%" in kpi:
        def fmt(v): return f"{v*100:.2f}%"
    elif "Revenue" in kpi or "Losses" in kpi:
        def fmt(v): return f"{v:,.0f} SAR"
    else:
        def fmt(v): return f"{v:,.1f}"

    lines = [
        f"TREND — {kpi}",
        f"Scope: BU={ctx.get('bu') or 'All'} | Year={ctx.get('year') or 'All'}",
        f"  Direction     : {direction}",
        f"  First month   : {monthly.iloc[0]['Month_Label']}  →  {fmt(first_val)}",
        f"  Last month    : {monthly.iloc[-1]['Month_Label']} →  {fmt(last_val)}",
        f"  Total change  : {fmt(change)} ({change_pct:+.1f}%)",
        f"  Best month    : {monthly.iloc[best_idx]['Month_Label']}  ({fmt(monthly.iloc[best_idx]['value'])})",
        f"  Worst month   : {monthly.iloc[worst_idx]['Month_Label']} ({fmt(monthly.iloc[worst_idx]['value'])})",
    ]
    return "\n".join(lines)


def investigate_kpi(model: OPDDataModel, kpi: str, ctx: dict[str, Any]) -> str:
    sections = [
        compute_gap_analysis(model, kpi, ctx),
        compute_driver_breakdown(model, kpi, ctx),
    ]
    if not ctx.get("doctor"):
        sections.append(compute_doctor_ranking(model, ctx, kpi))
    if not ctx.get("doctor") and not ctx.get("bu"):
        sections.append(compute_bu_comparison_text(model, ctx, kpi))
    sections.append(compute_trend_narrative(model, ctx, kpi))
    return "\n\n".join(sections)


def compute_summary(model: OPDDataModel, ctx: dict[str, Any]) -> dict[str, Any]:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return {}

    result: dict[str, Any] = {
        "rows": len(sub),
        "scope": {
            "bu": ctx.get("bu"),
            "doctor": ctx.get("doctor"),
            "year": ctx.get("year"),
        },
    }

    for col in model.kpi_columns:
        if col not in sub.columns:
            continue
        if col in SUM_KPIS:
            result[col] = {
                "agg": "total",
                "value": float(sub[col].sum()),
                "per_month": float(sub[col].mean()),
                "min_month": float(sub[col].min()),
                "max_month": float(sub[col].max()),
            }
        else:
            result[col] = {
                "agg": "average",
                "value": float(sub[col].mean()),
                "min": float(sub[col].min()),
                "max": float(sub[col].max()),
            }
    return result


def compute_bu_comparison(model: OPDDataModel, ctx: dict[str, Any]) -> pd.DataFrame:
    sub = model.filtered(year=ctx.get("year"))
    if ctx.get("bus"):
        sub = sub[sub["BU"].isin(ctx["bus"])]

    numeric_cols = [
        col for col in model.kpi_columns
        if col in sub.columns and col not in {"Year", "Month No"}
    ]
    agg_spec = {
        col: (col, "sum" if col in SUM_KPIS else "mean")
        for col in numeric_cols
    }
    grouped = sub.groupby("BU", as_index=False).agg(**agg_spec)

    if "Total Revenue" in grouped.columns and "Target Revenue" in grouped.columns:
        grouped["revenue_achievement_pct"] = grouped["Total Revenue"] / \
            grouped["Target Revenue"].replace(0, np.nan) * 100
        grouped["revenue_gap"] = grouped["Total Revenue"] - \
            grouped["Target Revenue"]
    if "No. Cases" in grouped.columns and "Target No. cases" in grouped.columns:
        grouped["cases_achievement_pct"] = grouped["No. Cases"] / \
            grouped["Target No. cases"].replace(0, np.nan) * 100
        grouped["cases_gap"] = grouped["No. Cases"] - \
            grouped["Target No. cases"]
    if "No. Booking" in grouped.columns and "No. Planned booking Slots" in grouped.columns:
        grouped["utilization"] = grouped["No. Booking"] / \
            grouped["No. Planned booking Slots"].replace(0, np.nan)
    if "Total Revenue" in grouped.columns and "No. Cases" in grouped.columns:
        grouped["realized_charge_per_case"] = grouped["Total Revenue"] / \
            grouped["No. Cases"].replace(0, np.nan)
    if "Total Revenue" in grouped.columns and "Total Leakage Revenue Losses" in grouped.columns:
        grouped["leakage_rate"] = grouped["Total Leakage Revenue Losses"] / \
            grouped["Total Revenue"].replace(0, np.nan)

    if "Patient Retention %" in grouped.columns:
        print("Patient Retention % found in grouped columns, creating 'retention' alias")
        grouped["retention"] = grouped["Patient Retention %"]
    if "Patient Acquisition %" in grouped.columns:
        grouped["acquisition"] = grouped["Patient Acquisition %"]
    if "Total Revenue" in grouped.columns:
        grouped["total_revenue"] = grouped["Total Revenue"]
    if "Target Revenue" in grouped.columns:
        grouped["target_revenue"] = grouped["Target Revenue"]
    if "No. Cases" in grouped.columns:
        grouped["no_cases"] = grouped["No. Cases"]
    if "Target No. cases" in grouped.columns:
        grouped["target_cases"] = grouped["Target No. cases"]

    sort_col = "Total Revenue" if "Total Revenue" in grouped.columns else (
        numeric_cols[0] if numeric_cols else None)
    if sort_col:
        return grouped.sort_values(sort_col, ascending=False)
    return grouped


def rank_doctors(model: OPDDataModel, ctx: dict[str, Any], kpi: str = "Total Revenue", top_n: int = 10) -> pd.DataFrame:
    sub = model.filtered(ctx.get("bu"), year=ctx.get("year"))
    if sub.empty or kpi not in sub.columns:
        return pd.DataFrame()
    agg = "mean" if "%" in kpi or kpi in {"Charge per case"} else "sum"
    grp = sub.groupby(["Doctor Name", "BU"],
                      as_index=False).agg(value=(kpi, agg))

    ascending = _resolve_rank_sorting(kpi, ctx.get("rank_preference"))
    return grp.sort_values("value", ascending=ascending)


def compute_trend(model: OPDDataModel, ctx: dict[str, Any], kpi: str) -> pd.DataFrame:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty or kpi not in sub.columns:
        return pd.DataFrame()
    agg = "mean" if "%" in kpi else "sum"
    return sub.groupby(["YearMonth", "BU"], as_index=False).agg(value=(kpi, agg)).sort_values("YearMonth")


def compute_forecast(model: OPDDataModel, ctx: dict[str, Any], kpi: str, periods: int = 3) -> pd.DataFrame:
    """Computes a simple linear forecast for the next N months using historical trend."""
    trend_df = compute_trend(model, ctx, kpi)
    if trend_df.empty or len(trend_df["YearMonth"].unique()) < 3:
        return pd.DataFrame()

    # Aggregate across BUs if multiple exist in the filtered set
    agg_trend = trend_df.groupby("YearMonth", as_index=False)[
        "value"].agg("sum" if kpi in SUM_KPIS else "mean")
    agg_trend = agg_trend.sort_values("YearMonth")

    # Use ordinal indices for regression fit
    x = np.arange(len(agg_trend))
    y = agg_trend["value"].values

    # Linear regression: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    # Project from the last available date
    last_date = pd.to_datetime(agg_trend["YearMonth"].iloc[-1] + "-01")

    forecast_rows = []
    for i in range(1, periods + 1):
        f_date = last_date + pd.DateOffset(months=i)
        f_ym = f_date.strftime("%Y-%m")
        val = slope * (len(agg_trend) + i - 1) + intercept
        forecast_rows.append(
            {"YearMonth": f_ym, "value": max(0, val), "Type": "Forecast"})

    agg_trend["Type"] = "Actual"
    return pd.concat([agg_trend, pd.DataFrame(forecast_rows)], ignore_index=True)


def data_context(model: OPDDataModel, question: str, ctx: dict[str, Any]) -> tuple[str, dict[str, pd.DataFrame]]:
    summary = compute_summary(model, ctx)
    bu_comp = compute_bu_comparison(model, ctx)
    kpi = ctx.get("kpi") or ("No-Show %" if "no-show" in question.lower()
                             or "no show" in question.lower() else "Total Revenue")
    
    # Resolve KPI with specific keyword fallback instead of a silent global default
    kpi = ctx.get("kpi")
    if not kpi:
        q_low = question.lower()
        if "no show" in q_low or "no-show" in q_low: kpi = "No-Show %"
        elif "revenue" in q_low: kpi = "Total Revenue"
        elif "acquisition" in q_low or "acuisision" in q_low: kpi = "Patient Acquisition %"
        else: kpi = "Total Revenue"

    doctors = rank_doctors(model, ctx, kpi=kpi)
    trend = compute_trend(model, ctx, kpi=kpi)

    tables = {"bu_comparison": bu_comp,
              "doctor_ranking": doctors, "trend": trend}
    lines = ["[LIVE DATA CONTEXT]"]

    # Compact mode is now only used for deep-dive investigations.
    # This allows the LLM to see structured rankings even when BU, Doctor, or Year filters are applied.
    compact_mode = is_investigation_query(question)

    if is_investigation_query(question):
        inv_kpi = detect_investigation_kpi(question) or kpi
        lines.append("[DYNAMIC INVESTIGATION]")
        lines.append(investigate_kpi(model, inv_kpi, ctx))

    wants_leakage_losses = any(
        key in question.lower()
        for key in ["revenue leakage", "leakage loss", "leakage losses", "total leakage"]
    )

    if summary:
        scope = summary.get("scope", {})
        lines.append("SUMMARY:")
        lines.append(
            f"Scope: BU={scope.get('bu') or 'All'} | Doctor={scope.get('doctor') or 'All'} | Year={scope.get('year') or 'All'}"
        )
        lines.append(f"Rows: {summary.get('rows', 0)} doctor-month records")
        summary_keys = [
            "Total Revenue",
            "Target Revenue",
            "No. Cases",
            "Target No. cases",
            "Charge per case",
            "Doctor PMS %",
            "No-Show %",
            "Service Leakage %",
            "Cross Referral %",
        ] if compact_mode else list(summary.keys())
        for key in summary_keys:
            value = summary.get(key)
            if key in {"rows", "scope"}:
                continue
            if not isinstance(value, dict):
                continue
            if key == "Total Leakage Revenue Losses" and not wants_leakage_losses:
                continue
            if value.get("agg") == "total":
                total = value.get("value", 0.0)
                per_month = value.get("per_month", 0.0)
                min_month = value.get("min_month", 0.0)
                max_month = value.get("max_month", 0.0)
                if "Revenue" in key or "Losses" in key:
                    lines.append(
                        f"- {key}: TOTAL={total:,.2f} SAR | avg per doctor-month={per_month:,.2f} SAR | "
                        f"lowest month={min_month:,.2f} SAR | highest month={max_month:,.2f} SAR"
                    )
                else:
                    lines.append(
                        f"- {key}: TOTAL={total:,.0f} | avg per doctor-month={per_month:,.1f} | "
                        f"min month={min_month:,.0f} | max month={max_month:,.0f}"
                    )
            else:
                avg_val = value.get("value", 0.0)
                min_val = value.get("min", 0.0)
                max_val = value.get("max", 0.0)
                if "%" in key:
                    lines.append(
                        f"- {key}: AVERAGE={avg_val*100:.2f}% | min={min_val*100:.2f}% | max={max_val*100:.2f}%"
                    )
                else:
                    lines.append(
                        f"- {key}: AVERAGE={avg_val:,.2f} | min={min_val:,.2f} | max={max_val:,.2f}"
                    )

    if not compact_mode and not bu_comp.empty:
        lines.append("\nBU COMPARISON:")
        lines.append(bu_comp.head(6).round(3).to_string(index=False))
        bu_kpi = ctx.get("kpi")
        bu_metric_col = _resolve_metric_column(bu_comp, bu_kpi)
        if bu_kpi and bu_metric_col:
            ranked_bu = bu_comp[["BU", bu_metric_col]].dropna().copy()
            if not ranked_bu.empty:
                ascending = _resolve_rank_sorting(
                    bu_kpi, ctx.get("rank_preference"))
                ranked_bu = ranked_bu.sort_values(
                    bu_metric_col, ascending=ascending).reset_index(drop=True)
                top_row = ranked_bu.iloc[0]
                top_bu = top_row["BU"]
                top_value = float(top_row[bu_metric_col])
                if "%" in bu_kpi:
                    top_value_str = f"{top_value*100:.2f}%"
                elif "Revenue" in bu_kpi or "Losses" in bu_kpi:
                    top_value_str = f"{top_value:,.0f} SAR"
                else:
                    top_value_str = f"{top_value:,.1f}"
                qualifier = "Best" if ctx.get("rank_preference") in {
                    "best", None} else "Top"
                lines.append(
                    f"{qualifier} BU for {bu_kpi}: {top_bu} — {top_value_str}")
    if not compact_mode and not doctors.empty:
        lines.append("\nDOCTOR RANKING:")
        lines.append(doctors.head(6).round(3).to_string(index=False))
        try:
            top = doctors.iloc[0]
            top_name = top.get("Doctor Name")
            top_val = float(top.get("value"))
            if "%" in kpi:
                top_val_str = f"{top_val*100:.2f}%"
            elif "Revenue" in kpi or "Losses" in kpi:
                top_val_str = f"{top_val:,.0f} SAR"
            else:
                top_val_str = f"{top_val:,.1f}"
            lines.append(f"Top doctor for {kpi}: {top_name} — {top_val_str}")
        except Exception:
            pass
    
    # Only include raw trend data in the prompt if the user is explicitly asking for a trend or forecast.
    is_trend_query = any(w in question.lower() for w in ["trend", "forecast", "predict", "projection", "time series", "over time", "history"])
    if not compact_mode and not trend.empty and is_trend_query:
        lines.append("\nRECENT TREND SAMPLE:")
        lines.append(trend.tail(18).round(3).to_string(index=False))

    # Add Time Series Forecast Context
    if ctx.get("chart_type") == "forecast" or any(w in question.lower() for w in ["forecast", "predict", "projection"]):
        f_kpi = ctx.get("kpi") or "Total Revenue"
        f_df = compute_forecast(model, ctx, f_kpi)
        if not f_df.empty:
            forecast_only_df = f_df[f_df["Type"] == "Forecast"]
            forecasts = forecast_only_df["value"].values
            actuals = f_df[f_df["Type"] == "Actual"]["value"].values

            # Determine direction
            direction = "INCREASING 📈" if forecasts[-1] > actuals[-1] else "DECREASING 📉"
            change_pct = ((forecasts[-1] - actuals[-1]) /
                          abs(actuals[-1]) * 100) if actuals[-1] != 0 else 0

            agg_forecast = forecast_only_df["value"].sum(
            ) if f_kpi in SUM_KPIS else forecast_only_df["value"].mean()
            agg_label = "Total" if f_kpi in SUM_KPIS else "Average"
            agg_str = f"{agg_forecast:,.0f} SAR" if ("Revenue" in f_kpi or "Losses" in f_kpi) else (
                f"{agg_forecast*100:.2f}%" if "%" in f_kpi else f"{agg_forecast:,.1f}")

            lines.append(f"\n[TIME SERIES FORECAST ANALYSIS — {f_kpi}]")
            lines.append(
                f"Forecasting Insight: The {f_kpi} is projected to follow a {direction} path over the next 3 months, "
                f"reaching a projected period {agg_label} of {agg_str}.")
            lines.append(f"Trend Direction: {direction}")
            lines.append(
                f"Projected change from last actual: {change_pct:+.1f}%")
            for _, row in forecast_only_df.iterrows():
                val = row["value"]
                val_str = f"{val:,.0f} SAR" if ("Revenue" in f_kpi or "Losses" in f_kpi) else (
                    f"{val*100:.2f}%" if "%" in f_kpi else f"{val:,.1f}")
                lines.append(f"- Projected for {row['YearMonth']}: {val_str}")

    return "\n".join(lines), tables


def build_driver_cards(model: OPDDataModel, ctx: dict[str, Any], kpi: str) -> go.Figure | None:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty or kpi not in KPI_DRIVER_GRAPH:
        return None

    drivers = KPI_DRIVER_GRAPH[kpi]["drivers"]
    direction = KPI_DRIVER_GRAPH[kpi]["direction"]
    card_items: list[dict[str, Any]] = []

    for drv in drivers:
        if drv not in sub.columns:
            continue
        actual = _agg_value(sub, drv)
        if np.isnan(actual):
            continue

        target = None
        has_target = False
        if drv == "No. Booking" and "No. Planned booking Slots" in sub.columns:
            target = float(sub["No. Planned booking Slots"].sum())
            has_target = True
        elif drv == "No. Cases" and "Target No. cases" in sub.columns:
            target = float(sub["Target No. cases"].sum())
            has_target = True

        is_bad = False
        gap_pct = 0.0
        if has_target and target is not None:
            if direction.get(drv, "low") == "low":
                is_bad = actual < target
            else:
                is_bad = actual > target
            gap_pct = (actual - target) / target * 100 if target != 0 else 0

        severity = _severity(gap_pct, is_bad)

        if "%" in drv:
            display_val = actual * 100
            number_fmt = ".2f"
            suffix = "%"
        elif drv in SUM_KPIS and ("Revenue" in drv or "Losses" in drv):
            display_val = actual
            number_fmt = ",.0f"
            suffix = " SAR"
        else:
            display_val = actual
            number_fmt = ",.0f"
            suffix = ""

        delta = None
        if has_target and target is not None:
            if "%" in drv:
                delta = {"reference": target * 100,
                         "valueformat": ".2f", "suffix": "%"}
            else:
                delta = {"reference": target, "valueformat": number_fmt}

        card_items.append({
            "title": f"{severity} {drv}",
            "value": display_val,
            "valueformat": number_fmt,
            "suffix": suffix,
            "delta": delta,
        })

    if not card_items:
        return None

    cols = 3
    rows = (len(card_items) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}
                for _ in range(cols)] for _ in range(rows)],
    )

    for idx, item in enumerate(card_items):
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(
            go.Indicator(
                mode="number+delta" if item["delta"] else "number",
                value=item["value"],
                number={
                    "valueformat": item["valueformat"], "suffix": item["suffix"]},
                title={"text": item["title"], "font": {"size": 12}},
                delta=item["delta"] or {},
            ),
            row=row,
            col=col,
        )

    scope = f"BU={ctx.get('bu') or 'All'} | Doctor={ctx.get('doctor') or 'All'} | Year={ctx.get('year') or 'All'}"
    fig.update_layout(
        title_text=f"Driver KPI Cards — {kpi} ({scope})",
        height=220 * rows,
        margin=dict(t=60, b=30, l=20, r=20),
    )
    return fig


def build_scope_cards(model: OPDDataModel, ctx: dict[str, Any]) -> go.Figure | None:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return None

    summary = compute_summary(model, ctx)
    if not summary:
        return None

    scope_title = "BU" if ctx.get(
        "bu") else "Doctor" if ctx.get("doctor") else "Scope"
    scope_value = ctx.get("bu") or ctx.get("doctor") or "All"
    title_prefix = f"{scope_title} Performance — {scope_value}"
    if ctx.get("year"):
        title_prefix += f" ({ctx['year']})"

    card_specs: list[dict[str, Any]] = []

    def add_card(label: str, value: float, *, is_percent: bool = False, is_sar: bool = False, delta: float | None = None) -> None:
        card_specs.append({
            "label": label,
            "value": value,
            "is_percent": is_percent,
            "is_sar": is_sar,
            "delta": delta,
        })

    total_revenue = summary.get("Total Revenue", {}).get("value")
    target_revenue = summary.get("Target Revenue", {}).get("value")
    no_cases = summary.get("No. Cases", {}).get("value")
    target_cases = summary.get("Target No. cases", {}).get("value")

    if isinstance(total_revenue, (int, float)):
        add_card("Total Revenue", total_revenue, is_sar=True,
                 delta=target_revenue if isinstance(target_revenue, (int, float)) else None)
    if isinstance(no_cases, (int, float)):
        add_card("No. Cases", no_cases,
                 delta=target_cases if isinstance(target_cases, (int, float)) else None)

    for metric in [
        "Charge per case",
        "No-Show %",
        "Patient Retention %",
        "Service Leakage %",
        "Cross Referral %",
    ]:
        metric_summary = summary.get(metric)
        if not metric_summary:
            continue
        value = metric_summary.get("value")
        if not isinstance(value, (int, float)):
            continue
        add_card(metric, value, is_percent="%" in metric)

    if not card_specs:
        return None

    cols = 3
    rows = (len(card_specs) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}
                for _ in range(cols)] for _ in range(rows)],
    )

    for idx, item in enumerate(card_specs):
        row = idx // cols + 1
        col = idx % cols + 1
        value = item["value"]
        number_format = ",.0f"
        suffix = ""
        delta = None

        if item["is_percent"]:
            value = value * 100
            number_format = ".2f"
            suffix = "%"
            if isinstance(item["delta"], (int, float)):
                delta = {"reference": item["delta"] * 100,
                         "valueformat": ".2f", "suffix": "%"}
        elif item["is_sar"]:
            number_format = ",.0f"
            suffix = " SAR"
            if isinstance(item["delta"], (int, float)):
                delta = {"reference": item["delta"],
                         "valueformat": ",.0f", "suffix": " SAR"}
        elif isinstance(item["delta"], (int, float)):
            delta = {"reference": item["delta"], "valueformat": ",.0f"}

        fig.add_trace(
            go.Indicator(
                mode="number+delta" if delta else "number",
                value=value,
                number={"valueformat": number_format, "suffix": suffix},
                title={"text": item["label"], "font": {"size": 12}},
                delta=delta or {},
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text=title_prefix,
        height=220 * rows,
        margin=dict(t=60, b=30, l=20, r=20),
    )
    return fig


def is_executive_dashboard_request(question: str) -> bool:
    q = question.lower()
    keywords = [
        "executive summary",
        "overall performance",
        "overall summary",
        "performance overview",
        "business overview",
        "executive overview",
        "summary of performance",
        "opd performance",
        "kpi snapshot",
        "dashboard",
        "management summary",
        "high level summary",
        "high-level summary",
        "monthly summary",
        "annual summary",
        "yearly summary",
        "review the performance",
    ]
    return any(keyword in q for keyword in keywords)


def build_metric_cards(model: OPDDataModel, ctx: dict[str, Any]) -> list[go.Figure]:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return []

    cards = []

    # Patient Retention %
    retention_avg = sub["Patient Retention %"].mean(
    ) * 100 if "Patient Retention %" in sub.columns else 0
    cards.append(
        go.Figure(data=[go.Indicator(
            mode="number",
            value=retention_avg,
            title={"text": "Avg Patient Retention %"},
            domain={"x": [0, 1], "y": [0, 1]}
        )])
    )

    # Avg No. Missed Opportunity
    missed_opp_avg = sub["No. Missed Opportunity"].mean(
    ) if "No. Missed Opportunity" in sub.columns else 0
    cards.append(
        go.Figure(data=[go.Indicator(
            mode="number",
            value=missed_opp_avg,
            title={"text": "Avg No. Missed Opportunity"},
            domain={"x": [0, 1], "y": [0, 1]}
        )])
    )

    # Avg Doctor PMS %
    pms_avg = sub["Doctor PMS %"].mean(
    ) * 100 if "Doctor PMS %" in sub.columns else 0
    cards.append(
        go.Figure(data=[go.Indicator(
            mode="number",
            value=pms_avg,
            title={"text": "Avg Doctor PMS %"},
            domain={"x": [0, 1], "y": [0, 1]}
        )])
    )

    return cards


def build_executive_dashboard(model: OPDDataModel, ctx: dict[str, Any]) -> list[go.Figure]:
    sub = model.filtered(ctx.get("bu"), ctx.get("doctor"), ctx.get("year"))
    if sub.empty:
        return []

    bu_comp = compute_bu_comparison(model, ctx)
    doctors = rank_doctors(model, ctx, kpi=ctx.get("kpi")
                           or "Total Revenue", top_n=10)
    trend_kpi = ctx.get("kpi") or "Total Revenue"
    trend = compute_trend(model, ctx, kpi=trend_kpi)

    charts: list[go.Figure] = []
    scope = f"BU={ctx.get('bu') or 'All'} | Doctor={ctx.get('doctor') or 'All'} | Year={ctx.get('year') or 'All'}"

    if ctx.get("doctor"):
        doctor_monthly = sub.groupby(["YearMonth"], as_index=False).agg(
            total_revenue=("Total Revenue", "sum"),
            no_cases=("No. Cases", "sum"),
            no_show=("No-Show %", "mean"),
            retention=("Patient Retention %", "mean"),
        )
        if not doctor_monthly.empty:
            charts.append(
                px.line(
                    doctor_monthly,
                    x="YearMonth",
                    y="total_revenue",
                    markers=True,
                    title=f"Total Revenue Trend — {scope}",
                )
            )
            charts.append(
                px.bar(
                    doctor_monthly,
                    x="YearMonth",
                    y="no_cases",
                    title=f"Monthly Cases — {scope}",
                )
            )
    elif ctx.get("bu"):
        bu_monthly = sub.groupby(["YearMonth"], as_index=False).agg(
            total_revenue=("Total Revenue", "sum"),
            target_revenue=("Target Revenue", "sum"),
            cases=("No. Cases", "sum"),
            target_cases=("Target No. cases", "sum"),
            no_show=("No-Show %", "mean"),
            retention=("Patient Retention %", "mean"),
        )
        if not bu_monthly.empty:
            charts.append(
                px.line(
                    bu_monthly,
                    x="YearMonth",
                    y="total_revenue",
                    markers=True,
                    title=f"Revenue Trend — {scope}",
                )
            )
            charts.append(
                px.bar(
                    bu_monthly,
                    x="YearMonth",
                    y=["cases", "target_cases"],
                    barmode="group",
                    title=f"Cases vs Target — {scope}",
                )
            )
    else:
        if not bu_comp.empty:
            bu_monthly = sub.groupby(["YearMonth", "BU"], as_index=False).agg(
                total_revenue=("Total Revenue", "sum")
            )
            if not bu_monthly.empty:
                charts.append(
                    px.line(
                        bu_monthly,
                        x="YearMonth",
                        y="total_revenue",
                        color="BU",
                        markers=True,
                        title=f"BU Revenue Trend — {scope}",
                    )
                )

            charts.append(
                px.bar(
                    bu_comp,
                    x="BU",
                    y=["revenue_achievement_pct", "cases_achievement_pct"],
                    barmode="group",
                    title="Revenue and Cases Achievement %",
                )
            )

    if not doctors.empty:
        doctors_plot = doctors.copy()
        if "%" in (ctx.get("kpi") or ""):
            doctors_plot["value"] = doctors_plot["value"] * 100
            title = f"{trend_kpi} Doctor Ranking (%)"
        else:
            title = f"{trend_kpi} Doctor Ranking"
        charts.append(
            px.bar(
                doctors_plot,
                x="Doctor Name",
                y="value",
                color="BU",
                title=title,
            )
        )

    if not ctx.get("bu") and not ctx.get("doctor"):
        metric_cards = build_metric_cards(model, ctx)
        charts.extend(metric_cards)
    elif not trend.empty:
        trend_plot = trend.copy()
        if "%" in trend_kpi:
            trend_plot["value"] = trend_plot["value"] * 100
            title = f"{trend_kpi} Trend (%)"
        else:
            title = f"{trend_kpi} Trend"
        charts.append(
            px.line(
                trend_plot,
                x="YearMonth",
                y="value",
                color="BU",
                markers=True,
                title=title,
            )
        )

    return charts[:6]


def build_charts(model: OPDDataModel, question: str, ctx: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[go.Figure]:
    q = question.lower()
    charts: list[go.Figure] = []
    bu_comp = tables.get("bu_comparison", pd.DataFrame())
    doctors = tables.get("doctor_ranking", pd.DataFrame())
    trend = tables.get("trend", pd.DataFrame())

    if is_executive_dashboard_request(question):
        dashboard_charts = build_executive_dashboard(model, ctx)
        if dashboard_charts:
            return dashboard_charts

    if is_investigation_query(question):
        inv_kpi = detect_investigation_kpi(question) or ctx.get("kpi")
        if inv_kpi and inv_kpi in KPI_DRIVER_GRAPH:
            card_fig = build_driver_cards(model, ctx, inv_kpi)
            if card_fig is not None:
                charts.append(card_fig)
                return charts

    single_scope = bool(ctx.get("bu") or ctx.get("doctor")
                        ) and len(ctx.get("bus") or []) <= 1
    # Include forecast in trend_query to skip indicator cards and show the line chart
    trend_query = "trend" in q or "trend analysis" in q or "time series" in q or ctx.get(
        "chart_type") == "forecast"
    # Detect if user wants a breakdown by doctor within their assigned scope
    is_doctor_breakdown = "doctor" in q and not ctx.get("doctor")

    if single_scope and not trend_query and not is_doctor_breakdown:
        main_kpi = ctx.get("kpi") or "Total Revenue"
        if "revenue" in main_kpi.lower():
            support_kpis = ["No. Cases", "Charge per case"]
        elif "%" in main_kpi:
            support_kpis = ["Doctor PMS %", "Patient Retention %"]
        else:
            support_kpis = ["No. Cases", "Doctor PMS %"]
        kpis = [main_kpi] + support_kpis[:2]
        summary = compute_summary(model, ctx) or {}
        card_items: list[dict[str, Any]] = []
        for k in kpis:
            item = summary.get(k)
            if item and isinstance(item.get("value"), (int, float)):
                val = item.get("value")
            else:
                val = 0
            card_items.append({
                "label": k,
                "value": val,
                "is_percent": ("%" in k),
                "is_sar": ("Revenue" in k or "Losses" in k),
            })
        cols = 3
        rows = 1
        fig = make_subplots(rows=rows, cols=cols,
                            specs=[[{"type": "indicator"} for _ in range(cols)]])
        for idx, item in enumerate(card_items):
            col = idx % cols + 1
            value = item["value"]
            number_format = ",.0f"
            suffix = ""
            if item["is_percent"]:
                value = value * 100
                number_format = ".2f"
                suffix = "%"
            elif item["is_sar"]:
                number_format = ",.0f"
                suffix = " SAR"
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    number={"valueformat": number_format, "suffix": suffix},
                    title={"text": item["label"], "font": {"size": 12}},
                ),
                row=1, col=col,
            )
        scope = f"BU={ctx.get('bu') or 'All'} | Doctor={ctx.get('doctor') or 'All'} | Year={ctx.get('year') or 'All'}"
        fig.update_layout(title_text=f"{main_kpi} — {scope}", height=220, margin=dict(
            t=50, b=20, l=20, r=20))
        charts.append(fig)
        return charts

    if "heatmap" in q:
        sub = model.filtered(ctx.get("bu"), year=ctx.get("year"))
        corr = sub[model.kpi_columns].corr(numeric_only=True)
        charts.append(
            px.imshow(corr, title="KPI Correlation Heatmap", aspect="auto"))
    elif "compare" in q and ctx.get("kpi") and not bu_comp.empty:
        kpi = ctx.get("kpi")
        y_col = _resolve_metric_column(bu_comp, kpi)
        if y_col is None:
            y_col = "Total Revenue" if "Total Revenue" in bu_comp.columns else None
        if y_col is None:
            return charts
        comp_plot = bu_comp.copy()
        if "%" in kpi:
            comp_plot[y_col] = comp_plot[y_col] * 100
            title = f"{kpi} by BU (%)"
        else:
            title = f"{kpi} by BU"
        charts.append(px.bar(comp_plot, x="BU", y=y_col, title=title))
    elif (ctx.get("chart_type") == "forecast" or "forecast" in q) and ctx.get("kpi"):
        kpi = ctx.get("kpi")
        forecast_df = compute_forecast(model, ctx, kpi)
        if not forecast_df.empty:
            fig = px.line(
                forecast_df,
                x="YearMonth",
                y="value",
                color="Type",
                line_dash="Type",
                markers=True,
                title=f"3-Month Linear Forecast — {kpi} ({ctx.get('bu') or 'All'})",
                template="plotly_white"
            )
            charts.append(fig)
    elif "trend" in q and not trend.empty:
        trend_plot = trend.copy()
        if "%" in (ctx.get("kpi") or ""):
            trend_plot["value"] = trend_plot["value"] * 100
            title = f"{ctx.get('kpi')} Trend (%)"
        else:
            title = f"{ctx.get('kpi') or 'KPI'} Trend"
        charts.append(
            px.line(
                trend_plot,
                x="YearMonth",
                y="value",
                color="BU",
                markers=True,
                title=title,
            )
        )
    elif ("doctor" in q or "scorecard" in q) and not doctors.empty:
        doctors_plot = doctors.copy()
        if "%" in (ctx.get("kpi") or ""):
            doctors_plot["value"] = doctors_plot["value"] * 100
            title = "Doctor Ranking (%)"
        else:
            title = "Doctor Ranking"
        charts.append(
            px.bar(
                doctors_plot,
                x="Doctor Name",
                y="value",
                color="BU",
                title=title,
            )
        )
    elif "dashboard" in q and not bu_comp.empty:
        charts.append(px.bar(bu_comp, x="BU", y="total_revenue",
                             title="Total Revenue by BU", text_auto=".2s"))
        charts.append(
            px.bar(
                bu_comp,
                x="BU",
                y=["revenue_achievement_pct", "cases_achievement_pct"],
                barmode="group",
                title="Revenue and Cases Achievement %",
            )
        )
    elif not bu_comp.empty:
        metric_col = _resolve_metric_column(
            bu_comp, ctx.get("kpi")) or "total_revenue"
        if metric_col not in bu_comp.columns:
            metric_col = "Total Revenue" if "Total Revenue" in bu_comp.columns else None
        if metric_col is not None:
            chart_title = f"{ctx.get('kpi') or 'Total Revenue'} by BU"
            if metric_col in {"Total Revenue", "total_revenue"}:
                chart_title = "Total Revenue by BU"
            charts.append(px.bar(bu_comp, x="BU", y=metric_col,
                                 title=chart_title, text_auto=".2s"))
    return charts


class OPDHealthcareAgent:
    def __init__(self, model: OPDDataModel, llm: GroqLLM, hr_db: EmployeeHRDB | None = None, user_email: str | None = None) -> None:
        self.model = model
        self.llm = llm
        self.hr_db = hr_db
        self.user_profile = None
        self.history: list[dict[str, str]] = []
        self.memory_manager = ConversationManager(llm, user_email=user_email)

        if self.hr_db and user_email:
            try:
                print(f"[HR_DB] Initializing session for: {user_email}...")
                self.user_profile = self.hr_db.get_user_profile(user_email)
                # Update memory manager with BU once profile is loaded
                if self.user_profile and self.user_profile.get("bu"):
                    self.memory_manager.bu = self.user_profile["bu"]
            except Exception as e:
                print(f"[HR_DB] Failed to fetch profile during init: {e}")

    def _prepare_missing_kpi_payload(self, question: str, ctx: dict[str, Any]) -> dict[str, Any] | None:
        """
        Prepares the payload for requesting a missing KPI to be added.
        """
        if not self.hr_db or not self.user_profile:
            print(
                "[MissingKPI] HR DB or user profile not available. Cannot prepare payload.")
            return None

        # Extract potential KPI name from the question
        # Simple heuristic: look for capitalized words or phrases, or just use the question
        kpi_name_match = re.search(
            r"(?:kpi|metric|measure|indicator)\s+([\w\s%#.-]+)", question, re.IGNORECASE)
        if kpi_name_match:
            kpi_name = kpi_name_match.group(1).strip()
        else:
            # Clean the question to extract a concise KPI title
            kpi_name = question.lower()
            for stop in ["what is", "tell me about", "show me", "can we calculate", "can you calculate", "?", "the"]:
                kpi_name = kpi_name.replace(stop, "")
            kpi_name = kpi_name.strip().title()
            if len(kpi_name) > 60:
                kpi_name = kpi_name[:57] + "..."

        if not kpi_name:
            kpi_name = "New KPI Request"

        requested_by_email = self.user_profile.get(
            "hr_email") or self.user_profile.get("hr_useremail", "")
        requested_by_name = self.user_profile.get(
            "hr_fullname", "Unknown User")
        bu_name = self.user_profile.get("bu") or ctx.get("bu") or "Unknown BU"

        if not requested_by_email:
            return None

        # Default assignee to the user specified in your working Postman example
        assignee_email = os.getenv(
            "KPI_REQUEST_ASSIGNEE_EMAIL", "Khaled.Arafa@Andalusiagroup.net")

        return {
            "KPI_Title": kpi_name,
            "KPI_Description": f"Request to add '{kpi_name}'. Original Query: {question}",
            "Justification": f"Requested by {requested_by_name} for Business Unit {bu_name}.",
            "KPI_Escalate": assignee_email,
            "KPI_Owner_Email": assignee_email,
            "Formula": "TBD",
            "raised_by": requested_by_email
        }

    def _prepare_missing_entity_payload(self, entity_name: str) -> dict[str, Any] | None:
        """Prepares the payload for requesting a missing data entity."""
        if not self.user_profile:
            return None

        requested_by_email = self.user_profile.get(
            "hr_email") or self.user_profile.get("hr_useremail", "")
        return {
            "Entity_Name": entity_name,
            "Business_Need": f"User is asking for data regarding '{entity_name}' which is missing from the current dataset.",
            "Raised_By_Email": requested_by_email,
            "Data_Type": ""  # To be filled via UI dropdown in app.py
        }

    def _prepare_escalation_payload(self, question: str, answer: str, ctx: dict[str, Any]) -> dict[str, Any] | None:
        """Detects if an escalation is suggested and constructs the Power Automate payload."""
        if not self.hr_db or not self.user_profile:
            return None

        # 1. Detection: Check if the user is asking to escalate or if the answer identifies underperformance
        combined_text = (question + " " + answer).lower()
        # Expanded triggers to catch data-driven indicators and performance gaps identified in the answer
        print(
            f"[Escalation] Evaluating text for triggers: '{combined_text[:100]}...'")
        triggers = [
            "escalate", "underperforming", "under target", "urgent", "critical", "gap", "root cause",
            "below target", "missed target", "declining", "worsening", "below the threshold",
            "improvement needed", "risk", "unacceptable", "red flag", "🔴", "🟠"
        ]

        is_triggered = any(t in combined_text for t in triggers)
        # Specifically catch playbook threshold violations (e.g., "below the 80% threshold")
        if not is_triggered and "below" in combined_text and "threshold" in combined_text:
            is_triggered = True

        if not is_triggered:
            print("[Escalation] No escalation triggers detected.")
            return None

        # 2. Extract KPI and BU context
        kpi = ctx.get("kpi") or detect_investigation_kpi(question)
        bu = ctx.get("bu")
        if not bu and self.user_profile:
            bu = self.user_profile.get("bu")
        print(f"[Escalation] Escalation triggered. KPI: {kpi}, BU: {bu}")

        if not kpi or not bu or str(bu).upper() in ["ALL", "CORPORATE", "GROUP"]:
            print(
                f"[Escalation] Aborting: Missing KPI or broad BU scope ({bu}).")
            return None

        # 3. Lookup Escalation Job Title from Knowledge Base
        escalation_title = None
        if not self.model.kb_map.empty:
            match = self.model.kb_map[self.model.kb_map["KPI_Name"].str.contains(
                kpi, case=False, na=False)]
            if not match.empty:
                # Prefer Action_Owner or Escalation_Level
                escalation_title = match.iloc[0].get(
                    "Action_Owner") or match.iloc[0].get("Escalation_Level")

        # If not found in KB, try to extract from the answer itself (if agent was prompted to include it)
        if not escalation_title or str(escalation_title).lower() == "nan":
            answer_lower = answer.lower()
            # Look for common management roles mentioned in the context of action/escalation
            role_match = re.search(
                r"(?:escalate to|notify|review by)\s+(medical manager|opd manager|operational manager|quality manager|hr business partner)", answer_lower)
            if role_match:
                escalation_title = role_match.group(
                    1).title()  # Capitalize for consistency
                print(
                    f"[Escalation] Extracted escalation title from answer: {escalation_title}")

        if not escalation_title or not isinstance(escalation_title, str) or str(escalation_title).lower() == "nan":
            print(
                f"[Escalation] No valid escalation title found for KPI: {kpi}. Skipping HR lookup.")
            return None

        # 4. Fetch the contact email using the position and BU
        print(
            f"[Escalation] Looking up user for job title: '{escalation_title}' in BU: '{bu}'")
        esc_user = self.hr_db.get_user_by_position_and_bu(
            str(escalation_title).strip(), bu)
        if not esc_user:
            print(
                f"[Escalation] No user found in HR DB for job title: '{escalation_title}' in BU: '{bu}'.")
            return None

        target_email = esc_user.get("hr_useremail")
        if not target_email:
            print(
                f"[Escalation] Found user {esc_user.get('hr_fullname')} but email is missing.")
            return None

        # 5. Build Payload
        print(
            f"[Escalation] Found escalation user: {esc_user.get('hr_fullname')} ({target_email}). Building payload.")
        user_email = self.user_profile.get(
            "hr_email") or self.user_profile.get("hr_useremail", "")
        return {
            "task_title": f"Performance Escalation: {kpi} in {bu}",
            "task_description": f"The Agent identified underperformance in {kpi}. Action suggested: {answer[:500]}...",
            "assignee_email": target_email,
            "manager_email": self.user_profile.get("hr_administrativemanagername") or "",
            "raised_by_email": user_email,
            "escalate_to_email": target_email,
            "due_date": (pd.Timestamp.now() + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
            "start_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "task_source": "OPD Medical Agent",
            "specialty": "",
            "bu_name": bu,
            "is_escalate": True,
            "escalation_path": f"{escalation_title} ({target_email})"
        }

    def ask(self, question: str) -> AgentReply:
        ctx = parse_query_context(question, self.model)

        # Enforce BU restriction based on User Profile
        if self.user_profile:
            assigned_bu = self.user_profile.get("bu")
            if assigned_bu:
                if assigned_bu in self.model.bus:
                    ctx["bu"] = assigned_bu
                    ctx["bus"] = [assigned_bu]
                elif assigned_bu in bu_scope.keys():
                    scope = bu_scope[assigned_bu]
                    if scope != "ALL":
                        ctx["bu"] = scope
                        ctx["bus"] = scope if isinstance(
                            scope, list) else [scope]
                    # If scope is "ALL", we don't apply profile-based filtering, allowing question-based filters to stay
                elif str(assigned_bu).upper() not in ["ALL", "CORPORATE", "GROUP", "ANDALUSIA GROUP", "AH", "AHQ"]:
                    ctx["bu"] = assigned_bu
                    ctx["bus"] = [assigned_bu]

        live_context, tables = data_context(self.model, question, ctx)
        charts = build_charts(self.model, question, ctx, tables)

        # Build Memory Context (Daily Summary + Accumulated KPIs)
        memory_context = self.memory_manager.build_memory_context(
            question, lookback_days=1)
        print(
            f"[ConversationMemory] Daily memory context loaded successfully for {self.memory_manager.user_email}")

        # Add policy RAG as supporting evidence, but never replace live OPD data.
        rag = _get_rag()
        policy_context = ""
        if rag:
            try:
                print("[PolicyRAG] Querying policy RAG for relevant context...")
                # Intent detection: If it looks like a policy query, search deeper and be more lenient
                is_policy = is_policy_query(question)
                n_to_fetch = 3 if is_policy else 2
                results = rag.query(question, n_results=n_to_fetch)

                # Results are already filtered by similarity_threshold inside rag.query()
                if results:
                    policy_context = rag.format_context(results)
                    policy_names = [r['policy_name'] for r in results]
                    print(f"[PolicyRAG] Policy context retrieved successfully ({len(results)} results): {', '.join(policy_names)}")
            except Exception as e:
                print(
                    f"[PolicyRAG] Query failed; continuing without policy context: {e}")

        history_context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}" for m in self.history[-6:])
        combined_context = live_context
        if policy_context:
            combined_context = f"{live_context}\n\n[POLICY CONTEXT]\n{policy_context}"

        kb = self.model.relevant_kb(question)
        chat_history_block = f"RECENT CHAT HISTORY:\n{history_context}\n\n" if history_context else ""
        # Note: We now rely more on the Memory Context in the System Prompt
        # but we can keep a short history for turn-by-turn continuity.
        prompt = (
            f"{chat_history_block}"
            f"{kb}\n\n"
            f"{combined_context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Answer using the live OPD data as primary evidence for performance metrics. "
            "If the question is about policies, rules, or operational standards (like working hours or thresholds), answer using the [POLICY CONTEXT] provided. "
            "If information is in both, prioritize live data values but explain them using the policy rules."
            "If the question is about policies, rules, or operational standards it's not missing data or KPIs it needs no triggers."
        )
        personalized_prompt = _get_user_based_system_prompt(
            SYSTEM_PROMPT, self.user_profile, memory_context=memory_context)

        answer, tool_calls = self.llm.generate(
            prompt, system_prompt_override=personalized_prompt, tools=AGENT_TOOLS)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Process Tool Calls (Missing Data Flows)
        missing_kpi_payload = None
        missing_entity_payload = None

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {}

            if func_name == "request_missing_kpi":
                kpi_arg = args.get("kpi_name", question)
                print(f"[ToolCall] Missing KPI detected: {kpi_arg}")
                missing_kpi_payload = self._prepare_missing_kpi_payload(
                    kpi_arg, ctx)

            elif func_name == "request_missing_entity":
                entity_arg = args.get("entity_name", "Unknown Entity")
                print(f"[ToolCall] Missing Entity detected: {entity_arg}")
                missing_entity_payload = self._prepare_missing_entity_payload(
                    entity_arg)

        # Save interaction (updates the daily record, regenerates summary, accumulates KPIs)
        self.memory_manager.save(question, answer, combined_context)

        # Check for potential escalation
        escalation_payload = self._prepare_escalation_payload(
            question, answer, ctx)

        return AgentReply(
            answer=answer,
            charts=charts,
            context=f"{kb}\n\n{combined_context}",
            tables=tables,
            escalation_payload=escalation_payload,
            missing_kpi_request_payload=missing_kpi_payload,
            missing_entity_payload=missing_entity_payload
        )
