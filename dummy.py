def parseContent(jdContent):

    title = None
    category = None
    description = None
    required_skills = None
    job_text = None

    for line in jdContent.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("Title:"):
            title = line.split("Title:", 1)[1].strip()

        elif line.startswith("category:"):
            category = line.split("category:", 1)[1].strip()

        elif line.startswith("description:"):
            description = line.split("description:", 1)[1].strip()

        elif line.startswith("required_skills:"):
            required_skills = line.split("required_skills:", 1)[1].strip()

        elif line.startswith("job_text:"):
            job_text = line.split("job_text:", 1)[1].strip()


    if title is None:
        title = "No Title Found"

    if category is None:
        category = "No Category Found"

    if description is None:
        description = "No Description Found"

    if required_skills is None:
        required_skills = "No Required Skills Found"

    if job_text is None:
        job_text = "No Job Text Found"


    jdData = {
        "title": title,
        "category": category,
        "description": description,
        "required_skills": required_skills,
        "job_text": job_text,
    }

    return jdData
    


resumeContent = """ 
Title: Backend Developer
category: Software Engineering
description: We are looking for a backend developer to build and maintain APIs and server-side applications.
required_skills: Python, FastAPI, SQL, Git, Docker
job_text: Backend Developer Software Engineering Python FastAPI SQL Git Docker
"""
resumeData = parseContent(resumeContent) 
for key, value in resumeData.items():
    print(f" {value}")