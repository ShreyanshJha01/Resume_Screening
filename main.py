import json
import os
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from candidate import analyze_resume, parseResume
from recruiter import analyze_job_description, parseContent

app = FastAPI(title="InstaHYR")
templates = Jinja2Templates(directory="templates")


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _build_match_summary(candidate_profile: dict, job_profile: dict) -> dict:
    candidate_skills = {str(skill).lower() for skill in _safe_list(candidate_profile.get("skills"))}
    job_skills = {str(skill).lower() for skill in _safe_list(job_profile.get("required_skills"))}
    matched = sorted(candidate_skills & job_skills)
    missing = sorted(job_skills - candidate_skills)
    score = round((len(matched) / len(job_skills) * 100), 1) if job_skills else 0
    summary = (
        f"This candidate matches {len(matched)} of {len(job_skills)} required skills for {job_profile.get('title', 'the role')}. "
        f"The strongest opportunity is to close the gap in {', '.join(missing[:3]) if missing else 'core role readiness'}."
    )
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "summary": summary,
    }


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/recruiter", response_class=HTMLResponse)
def recruiter_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="recruiter.html",
        context={"job_descriptions": [], "candidates": []},
    )


@app.post("/recruiter", response_class=HTMLResponse)
async def recruiter_portal(request: Request, jd: UploadFile | None = File(None)):
    job_descriptions = []
    candidates = []

    if jd is not None and getattr(jd, "filename", None):
        content = (await jd.read()).decode("utf-8", errors="ignore")
        jd_data = analyze_job_description(content)
        job_descriptions.append(jd_data)

    # Keep the older CSV-style behavior for compatibility when the user later hooks data storage in.
    if os.path.exists("job_descriptions.csv"):
        with open("job_descriptions.csv", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        job_descriptions.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

    if os.path.exists("candidates.csv"):
        with open("candidates.csv", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        candidates.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

    return templates.TemplateResponse(
        request=request,
        name="recruiter.html",
        context={"job_descriptions": job_descriptions, "candidates": candidates},
    )


@app.get("/candidate")
def candidate_portal(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="candidate.html",
        context={"analysis": {}, "match_summary": {}},
    )


@app.post("/candidate")
async def candidate_upload(request: Request, resume: UploadFile | None = File(None)):
    analysis = {}
    match_summary = {}

    if resume is not None and getattr(resume, "filename", None):
        content = (await resume.read()).decode("utf-8", errors="ignore")
        analysis = analyze_resume(content)

        if os.path.exists("job_descriptions.csv"):
            with open("job_descriptions.csv", "r", encoding="utf-8") as f:
                jobs = []
                for line in f:
                    if line.strip():
                        try:
                            jobs.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                if jobs:
                    job = jobs[0]
                    match_summary = _build_match_summary(analysis, job)

    return templates.TemplateResponse(
        request=request,
        name="candidate.html",
        context={"analysis": analysis, "match_summary": match_summary},
    )


@app.get("/health")
def health():
    return {"status": "ok", "app": "InstaHYR"}


__all__ = ["app", "analyze_resume", "analyze_job_description", "parseResume", "parseContent"]
