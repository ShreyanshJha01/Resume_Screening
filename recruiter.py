import json
import os
import re
from typing import Any

try:
    from google import genai
except Exception:  # pragma: no cover - safe fallback when SDK is missing.
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


def _fallback_job_analysis(job_text: str) -> dict:
    text = job_text or ""
    title_match = re.search(r"(?:Title|Role|Position)\s*[:\-]\s*([^\n]+)", text, flags=re.IGNORECASE)
    company_match = re.search(r"(?:Company|Organization)\s*[:\-]\s*([^\n]+)", text, flags=re.IGNORECASE)
    skills = []
    for word in [
        "python", "fastapi", "sql", "postgresql", "javascript", "typescript",
        "react", "node", "node.js", "docker", "aws", "azure", "git",
        "rest api", "machine learning", "data analysis", "excel", "java",
        "c++", "c#", "html", "css", "flask", "llm", "ai"
    ]:
        if word in text.lower():
            skills.append(word)
    return {
        "title": title_match.group(1).strip() if title_match else "Role",
        "company": company_match.group(1).strip() if company_match else "Company",
        "location": "",
        "job_type": "Full-time",
        "description": text[:400] if text else "No description provided.",
        "required_skills": sorted(set(skills)),
        "good_to_have": [],
        "summary": "Job description parsed locally because no Gemini API key was configured.",
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


def analyze_job_description(job_text: str) -> dict:
    clean_text = job_text or ""
    if not clean_text.strip():
        return {
            "title": "Untitled Role",
            "company": "Unknown Company",
            "location": "",
            "job_type": "Full-time",
            "description": "No job description was provided.",
            "required_skills": [],
            "good_to_have": [],
            "summary": "No job description was provided.",
        }

    prompt = (
        "You are a recruiter and ATS parser. Extract a clean JSON object from the following job description. "
        "Return only valid JSON with keys: title, company, location, job_type, description, required_skills, good_to_have, summary. "
        "Use shorter summary string and keep lists of skill names only. Do not include markdown fences.\n\n"
        f"JOB DESCRIPTION:\n{clean_text}"
    )

    try:
        parsed = _call_gemini(prompt)
        result = {
            "title": parsed.get("title") or "Untitled Role",
            "company": parsed.get("company") or "Unknown Company",
            "location": parsed.get("location") or "",
            "job_type": parsed.get("job_type") or "Full-time",
            "description": parsed.get("description") or clean_text[:1000],
            "required_skills": parsed.get("required_skills") or [],
            "good_to_have": parsed.get("good_to_have") or [],
            "summary": parsed.get("summary") or "Job description parsed with Gemini.",
        }
        if isinstance(result["required_skills"], str):
            result["required_skills"] = [s.strip() for s in result["required_skills"].split(",") if s.strip()]
        if isinstance(result["good_to_have"], str):
            result["good_to_have"] = [s.strip() for s in result["good_to_have"].split(",") if s.strip()]
        return result
    except Exception:
        return _fallback_job_analysis(clean_text)


def parseContent(jdContent):
    return analyze_job_description(jdContent)
