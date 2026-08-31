import re


def clean_resume_text(text):
    """
    Normalize resume text while preserving
    technical terms and useful information.
    """
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_job_text(text):
    """
    Normalize job description text.
    """
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_candidate_skill(skill):
    """
    Normalize an individual candidate skill.
    Removes experience duration such as '(3 years)'.
    """
    skill = str(skill).lower().strip()

    skill = re.sub(
        r"\s*\(\s*\d+(?:\.\d+)?\s*years?\s*\)",
        "",
        skill
    )

    skill = re.sub(r"\s+", " ", skill).strip()

    return skill


def normalize_candidate_skill_set(text):
    """
    Convert pipe-separated candidate skills
    into a normalized set.
    """
    skills = str(text).split("|")

    return {
        normalize_candidate_skill(skill)
        for skill in skills
        if str(skill).strip()
    }


def normalize_job_skill_set(text):
    """
    Convert comma-separated required job skills
    into a normalized set.
    """
    skills = str(text).split(",")

    return {
        skill.strip().lower()
        for skill in skills
        if skill.strip()
    }