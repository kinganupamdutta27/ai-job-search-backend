"""LangGraph Node — CV Analysis Agent.

Analyzes uploaded CV text using OpenAI to extract a structured profile
(skills, experience, preferred roles, etc.).
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from graph.models import CVProfile
from services.llm_utils import clean_llm_json, create_llm

logger = logging.getLogger(__name__)

CV_ANALYSIS_PROMPT = """You are an expert HR analyst and resume parser.

Analyze the following CV/resume text and extract a structured profile. Be thorough and accurate.

Return a JSON object with this exact schema:
{
  "name": "Full name of the candidate",
  "email": "Email address if found",
  "phone": "Phone number if found",
  "location": "City/Location if mentioned",
  "summary": "A 2-3 sentence professional summary based on the CV",
  "skills": ["skill1", "skill2", ...],
  "experience": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "duration": "e.g., Jan 2022 - Present",
      "description": "Brief description of role and achievements"
    }
  ],
  "education": [
    {
      "degree": "Degree name",
      "institution": "University/College",
      "year": "Graduation year or period"
    }
  ],
  "preferred_roles": ["Based on experience, list 3-5 job roles this person is best suited for"],
  "years_of_experience": 5.0
}

Important:
- For preferred_roles, infer from the most recent experience and skills
- If the person has GenAI/LLM experience, include roles like "GenAI Developer", "AI Engineer"
- If they have Python experience, include "Python Developer"
- Be specific with skills (e.g., "FastAPI" not just "Python frameworks")
- Calculate years_of_experience from the work history dates
- Return ONLY the JSON object, no markdown or extra text."""


async def analyze_cv(state: GraphState) -> dict:
    """
    LangGraph node: Analyze CV text and extract structured profile.

    Reads: state["cv_text"]
    Writes: state["cv_profile"], state["current_step"]
    """
    logger.info("📄 Starting CV analysis...")

    cv_text = state.get("cv_text", "")
    if not cv_text:
        return {
            "errors": ["No CV text provided for analysis"],
            "current_step": "failed",
        }

    llm = create_llm(temperature=0.1)

    messages = [
        SystemMessage(content=CV_ANALYSIS_PROMPT),
        HumanMessage(content=f"Here is the CV text to analyze:\n\n{cv_text}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = clean_llm_json(response.content)
        profile_data = json.loads(content)
        profile = CVProfile(**profile_data)

        logger.info(
            f"✅ CV analyzed: {profile.name}, "
            f"{len(profile.skills)} skills, "
            f"{len(profile.preferred_roles)} preferred roles"
        )

        return {
            "cv_profile": profile.model_dump(),
            "current_step": "cv_analyzed",
        }

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse CV analysis JSON: {e}")
        return {
            "errors": [f"CV analysis returned invalid JSON: {str(e)}"],
            "current_step": "failed",
        }
    except Exception as e:
        logger.error(f"❌ CV analysis failed: {e}")
        return {
            "errors": [f"CV analysis error: {str(e)}"],
            "current_step": "failed",
        }
