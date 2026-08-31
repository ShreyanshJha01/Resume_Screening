import json
import os
import re
from typing import Any

try:
    from google import genai
except Exception:  # pragma: no cover - this is a safe fallback path.
    genai = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_json(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    text = _clean_text(payload)
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _fallback_resume_analysis(resume_text: str) -> dict:
    text = resume_text or ""
    lowered = text.lower()
    skills = []
    for word in [
        "python", "fastapi", "sql", "postgresql", "javascript", "typescript",
        "react", "node", "node.js", "docker", "aws", "azure", "git",
        "rest api", "machine learning", "data analysis", "excel", "java",
        "c++", "c#", "html", "css", "flask", "llm", "ai"
    ]:
        if word in lowered:
            skills.append(word)
    skills = sorted(set(skills))

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text, flags=re.MULTILINE)
    experience_match = re.search(r"(\d+\.?\d*)\s*(?:years?|yrs?)", text, flags=re.IGNORECASE)

    return {
        "name": name_match.group(1) if name_match else "Candidate",
        "email": email_match.group(0) if email_match else "",
        "phone": "",
        "skills": skills,
        "experience_years": float(experience_match.group(1)) if experience_match else 0,
        "education": [segment.strip() for segment in re.findall(r"(?:Bachelor|Master|Diploma|B\.Tech|M\.Tech|Degree)[^\n]+", text, flags=re.IGNORECASE)][:5],
        "certifications": [segment.strip() for segment in re.findall(r"(?:AWS|Azure|Google|Cisco|PMP|Oracle|Scrum|Certified)[^\n]+", text, flags=re.IGNORECASE)][:5],
        "projects": [],
        "summary": "Profile parsed locally because no Gemini API key was configured. The candidate profile is ready for LLM-based refinement when configured.",
    }


def _call_gemini(prompt: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    if not api_key or genai is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 800,
            "response_mime_type": "application/json",
        },
    )
    return _safe_json(getattr(response, "text", response))


def analyze_resume(resume_text: str) -> dict:
    clean_text = resume_text or ""
    if not clean_text.strip():
        return {
            "name": "Candidate",
            "email": "",
            "phone": "",
            "skills": [],
            "experience_years": 0,
            "education": [],
            "certifications": [],
            "projects": [],
            "summary": "No resume content was provided.",
        }

    prompt = (
        "You are an ATS and resume parser. Extract a clean JSON object from the following resume text. "
        "Return only valid JSON with these keys: name, email, phone, skills, experience_years, education, certifications, projects, summary. "
        "Normalize skill names to common professional terms. Do not include markdown fences.\n\n"
        f"RESUME:\n{clean_text}"
    )

    try:
        parsed = _call_gemini(prompt)
        result = {
            "name": parsed.get("name") or "Candidate",
            "email": parsed.get("email") or "",
            "phone": parsed.get("phone") or "",
            "skills": parsed.get("skills") or [],
            "experience_years": parsed.get("experience_years") or 0,
            "education": parsed.get("education") or [],
            "certifications": parsed.get("certifications") or [],
            "projects": parsed.get("projects") or [],
            "summary": parsed.get("summary") or "Resume summary generated with Gemini.",
        }
        if result["skills"] and isinstance(result["skills"], str):
            result["skills"] = [s.strip() for s in result["skills"].split(",") if s.strip()]
        return result
    except Exception:
        return _fallback_resume_analysis(clean_text)


def parseResume(resumeContent):
    return analyze_resume(resumeContent)
