"""Centralized prompt builder for AI email generation.

Constructs prompts with role-specific tone, company context,
and customizable templates. Supports both job-specific and
proactive outreach modes.
"""

from __future__ import annotations

from graph.models import CVProfile, JobListing, HRContact


# ── Tone Presets ─────────────────────────────────────────────────────────────

TONE_PRESETS = {
    "formal": "Use formal, corporate language. Address recipient by title if known.",
    "friendly": "Be warm and approachable, yet professional. Show genuine enthusiasm.",
    "direct": "Be concise and action-oriented. Get to the point quickly.",
    "executive": "Use executive-level communication. Emphasize leadership and strategic value.",
    "professional": "Be professional and enthusiastic about the opportunity.",
}


def _get_tone_instruction(tone: str) -> str:
    """Get the tone instruction string for a given tone preset."""
    return TONE_PRESETS.get(tone.lower(), TONE_PRESETS["professional"])


def _get_recent_role(profile: CVProfile) -> str:
    """Extract the most recent role string from a profile."""
    if profile.experience:
        exp = profile.experience[0]
        return f"{exp.title} at {exp.company} ({exp.duration})"
    return "Not specified"


# ── Job-Specific Email Prompt ────────────────────────────────────────────────


def build_job_email_prompt(
    profile: CVProfile,
    listing: JobListing,
    contact: HRContact,
    tone: str = "professional",
    role_emphasis: str = "",
    company_context: str = "",
) -> str:
    """Build a prompt for generating a job-specific application email.

    Args:
        profile:         Candidate CV profile.
        listing:         Job listing to apply for.
        contact:         HR contact to address.
        tone:            One of: formal, friendly, direct, executive, professional.
        role_emphasis:    Role-specific skills to emphasize.
        company_context:  Additional context about the company (news, culture, etc.).

    Returns:
        Complete system prompt string for the LLM.
    """
    tone_instruction = _get_tone_instruction(tone)

    context_block = ""
    if company_context:
        context_block = f"""
Company Context (use this to personalize):
{company_context}
"""

    return f"""You are an expert at writing professional job application emails.

Write a personalized email for the following job application. The email should be:
- Professional but warm and personable
- Tailored to the specific job role and company
- Highlighting relevant skills from the candidate's profile
- {tone_instruction}
- Concise (max 200 words for body)

Candidate Profile:
- Name: {profile.name}
- Skills: {", ".join(profile.skills[:10])}
- Recent Role: {_get_recent_role(profile)}
- Years of Experience: {profile.years_of_experience}
- Summary: {profile.summary}

Job Details:
- Title: {listing.title}
- Company: {listing.company}
- Description: {listing.description_snippet[:300]}

HR Contact Name: {contact.name}
{context_block}
Role emphasis areas: {role_emphasis or "relevant technical skills and professional experience"}

Return a JSON object with:
{{
  "subject": "Compelling email subject line",
  "greeting": "Dear [Name]," or "Dear Hiring Manager,",
  "body": "Main email body (highlight 2-3 relevant skills/experiences)",
  "skills_highlight": "Comma-separated list of 4-5 most relevant skills for this role",
  "closing": "Professional closing line (e.g., 'Looking forward to discussing...')"
}}

Return ONLY the JSON object."""


# ── Proactive Outreach Prompt ────────────────────────────────────────────────


def build_proactive_email_prompt(
    profile: CVProfile,
    company: str,
    contact: HRContact,
    tone: str = "professional",
    company_context: str = "",
) -> str:
    """Build a prompt for generating a proactive cold-outreach email.

    Args:
        profile:         Candidate CV profile.
        company:         Target company name.
        contact:         HR contact to address.
        tone:            One of: formal, friendly, direct, executive, professional.
        company_context:  Additional context about the company.

    Returns:
        Complete system prompt string for the LLM.
    """
    tone_instruction = _get_tone_instruction(tone)

    context_block = ""
    if company_context:
        context_block = f"""
Company Context (use this to personalize):
{company_context}
"""

    return f"""You are an expert at writing professional cold-outreach emails for job seekers.

Write a proactive introduction email to a company asking about potential job openings.
This is NOT for a specific job posting — the candidate is proactively reaching out
to introduce themselves and inquire about relevant opportunities.

The email should be:
- Professional, confident, and concise
- Highlight the candidate's best skills and experience
- Ask about current or upcoming openings that match their profile
- {tone_instruction}
- Concise (max 200 words for body)

Candidate Profile:
- Name: {profile.name}
- Skills: {", ".join(profile.skills[:10])}
- Recent Role: {_get_recent_role(profile)}
- Years of Experience: {profile.years_of_experience}
- Summary: {profile.summary}

Target Company: {company}
HR Contact Name: {contact.name}
{context_block}
The candidate is looking for roles such as: {", ".join(profile.preferred_roles[:3])}

Return a JSON object with:
{{
  "subject": "Compelling subject line (e.g., 'Experienced [Role] — Exploring Opportunities at [Company]')",
  "greeting": "Dear [Name]," or "Dear Hiring Team,",
  "body": "Main body: introduce yourself, highlight top skills, ask about availability of matching roles",
  "skills_highlight": "Comma-separated list of 4-5 most relevant skills",
  "closing": "Professional closing (e.g., 'I would welcome the opportunity to discuss...')"
}}

Return ONLY the JSON object."""
