"""Email template management using Jinja2."""

from __future__ import annotations

from jinja2 import Environment, BaseLoader

# ── Base HTML Email Template ─────────────────────────────────────────────────

BASE_EMAIL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .greeting { margin-bottom: 16px; }
        .body-content { margin-bottom: 16px; }
        .skills-highlight {
            background: #f0f7ff;
            border-left: 4px solid #2563eb;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 0 4px 4px 0;
        }
        .closing { margin-top: 24px; }
        .signature {
            margin-top: 8px;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="greeting">{{ greeting }}</div>

    <div class="body-content">{{ body }}</div>

    {% if skills_highlight %}
    <div class="skills-highlight">
        <strong>Key Relevant Skills:</strong><br>
        {{ skills_highlight }}
    </div>
    {% endif %}

    <div class="closing">{{ closing }}</div>

    <div class="signature">
        {{ sender_name }}<br>
        {% if sender_email %}{{ sender_email }}<br>{% endif %}
        {% if sender_phone %}{{ sender_phone }}{% endif %}
    </div>
</body>
</html>"""

# ── Plain Text Template ──────────────────────────────────────────────────────

BASE_TEXT_TEMPLATE = """{{ greeting }}

{{ body }}

{% if skills_highlight %}Key Relevant Skills:
{{ skills_highlight }}
{% endif %}

{{ closing }}

{{ sender_name }}
{% if sender_email %}{{ sender_email }}{% endif %}
{% if sender_phone %}{{ sender_phone }}{% endif %}"""


# ── Role-Specific Content Guidance ───────────────────────────────────────────

ROLE_TEMPLATES: dict[str, dict[str, str]] = {
    "python_developer": {
        "emphasis": "Python development expertise including Django, FastAPI, Flask, data structures, algorithms, and automated testing",
        "tone": "technical and confident",
    },
    "genai_developer": {
        "emphasis": "Generative AI, LLMs, LangChain, LangGraph, RAG systems, prompt engineering, and AI application development",
        "tone": "innovative and forward-thinking",
    },
    "data_scientist": {
        "emphasis": "Machine learning, statistical analysis, data visualization, Python (pandas, scikit-learn, TensorFlow/PyTorch)",
        "tone": "analytical and data-driven",
    },
    "full_stack_developer": {
        "emphasis": "Full-stack development with React/Next.js, Node.js/Python backends, databases, REST APIs, and DevOps",
        "tone": "versatile and solution-oriented",
    },
    "devops_engineer": {
        "emphasis": "CI/CD pipelines, Docker, Kubernetes, cloud platforms (AWS/GCP/Azure), infrastructure as code",
        "tone": "reliability-focused and systematic",
    },
    "backend_developer": {
        "emphasis": "Server-side architecture, microservices, APIs, databases (SQL/NoSQL), caching, and scalability",
        "tone": "architectural and performance-minded",
    },
    "default": {
        "emphasis": "relevant technical skills and professional experience",
        "tone": "professional and enthusiastic",
    },
}


def get_role_template(role: str) -> dict[str, str]:
    """Get the role-specific template guidance for email generation."""
    role_key = role.lower().replace(" ", "_").replace("-", "_")
    # Try exact match, then partial match, then default
    if role_key in ROLE_TEMPLATES:
        return ROLE_TEMPLATES[role_key]
    for key, value in ROLE_TEMPLATES.items():
        if key in role_key or role_key in key:
            return value
    return ROLE_TEMPLATES["default"]


def render_email_html(
    greeting: str,
    body: str,
    closing: str,
    sender_name: str,
    sender_email: str = "",
    sender_phone: str = "",
    skills_highlight: str = "",
    custom_template: str | None = None,
) -> str:
    """Render the HTML email using Jinja2."""
    env = Environment(loader=BaseLoader())
    template_str = custom_template or BASE_EMAIL_TEMPLATE
    template = env.from_string(template_str)
    return template.render(
        greeting=greeting,
        body=body,
        closing=closing,
        sender_name=sender_name,
        sender_email=sender_email,
        sender_phone=sender_phone,
        skills_highlight=skills_highlight,
    )


def render_email_text(
    greeting: str,
    body: str,
    closing: str,
    sender_name: str,
    sender_email: str = "",
    sender_phone: str = "",
    skills_highlight: str = "",
) -> str:
    """Render the plain-text email using Jinja2."""
    env = Environment(loader=BaseLoader())
    template = env.from_string(BASE_TEXT_TEMPLATE)
    return template.render(
        greeting=greeting,
        body=body,
        closing=closing,
        sender_name=sender_name,
        sender_email=sender_email,
        sender_phone=sender_phone,
        skills_highlight=skills_highlight,
    )


# ── Template Storage (in-memory for now) ─────────────────────────────────────

_custom_templates: dict[str, str] = {
    "default": BASE_EMAIL_TEMPLATE,
}


def get_template(template_id: str = "default") -> str:
    """Get a stored email template by ID."""
    return _custom_templates.get(template_id, BASE_EMAIL_TEMPLATE)


def save_template(template_id: str, html_content: str) -> None:
    """Save a custom email template."""
    _custom_templates[template_id] = html_content


def list_templates() -> dict[str, str]:
    """List all available templates (id -> first 100 chars preview)."""
    return {tid: tmpl[:100] + "..." for tid, tmpl in _custom_templates.items()}
