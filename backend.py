import os
import re
import numpy as np
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from datetime import datetime, timezone
import uvicorn
from dotenv import load_dotenv

# os.environ['HF_HUB_DISABLE_XET'] = '1'
# # Optionally disable transfer for extra safety (helps with resume/chunk issues)
# os.environ['HF_HUB_DISABLE_TRANSFER'] = '1'

load_dotenv()
api_key = os.getenv("HF_TOKEN")

app = FastAPI()

# CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your Streamlit port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# Load embeddings model and both vector stores once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
job_vectorstore = FAISS.load_local(
    "vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True
)
resume_vectorstore = FAISS.load_local(
    "vector_store/resume_faiss", embeddings, allow_dangerous_deserialization=True
)




#  Extract text from uploaded PDF resume
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Clean the text data
def clean_text(text):
    """Cleans text by converting to lowercase, removing special characters,
    and handling whitespace."""
    try:
        text = str(text)
    except Exception as e:
        print(f"Error converting text to string: {e}")
        return text
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    text = re.sub(r"\t", r"", text).strip()
    text = re.sub(r"(\n|\r)+", r"\n", text).strip()
    text = re.sub(r" +", r" ", text).strip()
    return text


def get_llm_response(api_key, prompt, model="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Sends a prompt to Hugging Face Inference API and returns the generated text.
    """
    client = InferenceClient(provider= "featherless-ai", token=api_key)
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Page 1: Resume Details
@app.post("/resume_details")
async def resume_details(file: UploadFile = File(...), api_key: str = Form(...)):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    You are a resume parser. Extract and label ONLY these sections from the resume text below:
    - Name
    - Email
    - LinkedIn Profile
    - GitHub Profile
    - Portfolio
    - Phone Number
    - Education
    - Skills
    - Experience
    - Projects
    - Achievements
    - Certifications
    - Extra-curricular Activities

    Rules:
    - Output in bullet points: **Section Name**: Extracted content (keep concise).
    - Include ONLY sections present in the text—skip missing ones.
    - If no relevant text, output nothing for that section.
    - Use bold for section names.
    - Do NOT add any extra commentary or information.
    - Ensure the output is clean and easy to read.
    - If the resume is empty or unreadable, respond with "No content found in the resume."
    - The section headings should be in bold and big compared to the rest of the text.
    - Display the name in capitalized format.
    - Don't include any labels like "•" or "-".
    - Don't display the project github links.

    Resume Text:
    {resume_text[:4000]}  # Slightly longer limit

    Start output directly with bullets—no intro text.
    """

    feedback = get_llm_response(api_key, prompt)
    print(feedback)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 2: Resume Matching
@app.post("/resume_matching")
async def resume_matching(
    file: UploadFile = File(...), 
    job_description: str = Form(...), 
    api_key: str = Form(...)
):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    You are an expert resume evaluator. Analyze the resume against the job description.

    Job Description:
    {job_description}

    Resume:
    {resume_text[:3000]}

    You MUST start your response with exactly these score lines (each on its own line, with a number 0-100):
    TOTAL_MATCH_SCORE: <number>
    SKILLS_SCORE: <number>
    EXPERIENCE_SCORE: <number>
    EDUCATION_SCORE: <number>
    PROJECTS_SCORE: <number>

    Then provide detailed feedback using these exact section headings (use markdown ## for headings):

    ## Areas of Strength
    List the candidate's strongest matching qualifications as bullet points.

    ## Areas of Weakness
    List gaps or weaker areas as bullet points.

    ## Missing Skills & Qualifications
    List specific skills or qualifications from the job description that are missing from the resume.

    ## Matching Skills & Qualifications
    List skills and qualifications that match between the resume and job description.

    ## Suggestions for Improvement
    Provide specific, actionable suggestions to improve the resume match score.

    Rules:
    - Be concise and professional.
    - Use bullet points (- ) for items within each section.
    - Every section heading must use ## markdown format.
    - Scores should be realistic and well-justified.
    """
    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 3: Chat with Resume and Job Description
@app.post("/chat_with_resume")
async def chat_with_resume(
    query: str = Form(...),
    api_key: str = Form(...),
    job_description: str = Form(""),
    file: UploadFile = File(...)
):  
    # ── Extract and clean the user's uploaded resume ──
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # ── Build context from all sources ──
    context_sections = []

    # Always include the user's uploaded resume
    context_sections.append(
        f"--- Candidate's Resume (uploaded by user) ---\n{resume_text[:4000]}"
    )

    # Include job description if provided
    if job_description.strip():
        jd_cleaned = clean_text(job_description)
        context_sections.append(
            f"--- Job Description (provided by user) ---\n{jd_cleaned[:3000]}"
        )

    # Search vector stores for matching context
    resume_chunks = splitter.split_text(resume_text)
    if resume_chunks:
        chunk_embeddings = [embeddings.embed_query(chunk) for chunk in resume_chunks]
        resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    else:
        resume_embedding = embeddings.embed_query(resume_text[:2000])

    # Search for job descriptions that semantically match the resume's skills/content
    job_results = job_vectorstore.similarity_search_by_vector(resume_embedding, k=5)
    # Search for similar resumes from the global store
    resume_results = resume_vectorstore.similarity_search_by_vector(resume_embedding, k=3)

    if job_results:
        job_context = "\n\n".join([doc.page_content for doc in job_results])
        context_sections.append(
            f"--- Matching Job Descriptions (from database, matched to candidate's skills) ---\n{job_context}"
        )
    if resume_results:
        resume_context = "\n\n".join([doc.page_content for doc in resume_results])
        context_sections.append(
            f"--- Similar Resumes (from database, for comparison) ---\n{resume_context}"
        )

    combined_context = "\n\n".join(context_sections)

    # ── Create the LLM prompt ──
    prompt = f"""You are an intelligent career assistant. Answer the user's question based ONLY on the context provided below.

User's Question: {query}

{combined_context}

Instructions:
- Answer the question based strictly on the context above.
- When the user asks about the candidate, resume, skills, projects, education, or experience, refer ONLY to the "Candidate's Resume" section.
- When the user asks about matching job descriptions, refer to the "Matching Job Descriptions" section — these were found by matching the candidate's resume skills against a database of real job postings.
- Do NOT invent or assume information that is not present in the context.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and professional.
- When referring to the person whose resume was uploaded, call them "the candidate".
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# Page 3 helper: Save resume to vector store
@app.post("/save_resume_to_vectorstore")
async def save_resume_to_vectorstore(
    file: UploadFile = File(...),
):
    """Extract text from uploaded resume PDF, chunk it, and add to the global FAISS store."""
    global resume_vectorstore
    try:
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)

        if not resume_text.strip():
            return {"success": False, "message": "Could not extract text from the PDF."}

        chunks = splitter.split_text(resume_text)
        if not chunks:
            return {"success": False, "message": "No text chunks generated from the resume."}

        from langchain.schema import Document
        docs = [
            Document(page_content=chunk, metadata={"source": file.filename})
            for chunk in chunks
        ]

        resume_vectorstore.add_documents(docs)
        resume_vectorstore.save_local("vector_store/resume_faiss")

        return {
            "success": True,
            "message": f"Resume saved! {len(docs)} chunks added to vector store.",
        }
    except Exception as e:
        return {"success": False, "message": f"Error saving resume: {str(e)}"}


# Save job description text to vector store
@app.post("/save_jd_to_vectorstore")
async def save_jd_to_vectorstore(
    job_description: str = Form(...),
):
    """Clean, chunk, and add a job description to the global FAISS job store."""
    global job_vectorstore
    try:
        jd_text = clean_text(job_description)

        if not jd_text.strip():
            return {"success": False, "message": "Job description text is empty."}

        chunks = splitter.split_text(jd_text)
        if not chunks:
            return {"success": False, "message": "No text chunks generated from the job description."}

        from langchain.schema import Document
        docs = [
            Document(page_content=chunk, metadata={"source": "user_input"})
            for chunk in chunks
        ]

        job_vectorstore.add_documents(docs)
        job_vectorstore.save_local("vector_store/job_faiss")

        return {
            "success": True,
            "message": f"Job description saved! {len(docs)} chunks added to vector store.",
        }
    except Exception as e:
        return {"success": False, "message": f"Error saving job description: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════
# Compare Resumes
# ═══════════════════════════════════════════════════════════════════════
@app.post("/compare_resumes")
async def compare_resumes(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    api_key: str = Form(...)
):
    """Score each uploaded resume against the job description and return ranked results."""
    results = []

    for idx, file in enumerate(files, 1):
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)

        if not resume_text.strip():
            results.append({
                "filename": file.filename,
                "upload_order": idx,
                "total_score": 0,
                "skills": 0,
                "experience": 0,
                "education": 0,
                "projects": 0,
                "summary": "Could not extract text from this PDF.",
            })
            continue

        prompt = f"""You are an expert resume evaluator. Score this resume against the job description.

Job Description:
{job_description[:3000]}

Resume {idx} ({file.filename}):
{resume_text[:3000]}

You MUST start your response with exactly these score lines (each on its own line, number 0-100):
TOTAL_MATCH_SCORE: <number>
SKILLS_SCORE: <number>
EXPERIENCE_SCORE: <number>
EDUCATION_SCORE: <number>
PROJECTS_SCORE: <number>

Then write a SHORT 2-3 sentence summary of the candidate's fit for this role.
Keep the summary concise — no bullet points, no section headings.
"""

        feedback = get_llm_response(api_key, prompt)

        # Parse scores from response
        scores = {}
        for label, pattern in {
            "total_score": r"TOTAL_MATCH_SCORE[:\s]*(\d{1,3})",
            "skills": r"SKILLS_SCORE[:\s]*(\d{1,3})",
            "experience": r"EXPERIENCE_SCORE[:\s]*(\d{1,3})",
            "education": r"EDUCATION_SCORE[:\s]*(\d{1,3})",
            "projects": r"PROJECTS_SCORE[:\s]*(\d{1,3})",
        }.items():
            match = re.search(pattern, feedback, re.IGNORECASE)
            scores[label] = min(int(match.group(1)), 100) if match else 0

        # Extract summary (everything after the score lines)
        summary = re.sub(
            r"(?:TOTAL_MATCH_SCORE|SKILLS_SCORE|EXPERIENCE_SCORE|EDUCATION_SCORE|PROJECTS_SCORE)[:\s]*\d{1,3}[/\d]*\s*",
            "", feedback
        ).strip()

        results.append({
            "filename": file.filename,
            "upload_order": idx,
            **scores,
            "summary": summary,
        })

        # Reset file pointer for potential reuse
        file.file.seek(0)

    # Sort by total_score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)
    return {"results": results}


@app.post("/chat_with_comparison")
async def chat_with_comparison(
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    job_description: str = Form(...),
    api_key: str = Form(...)
):
    """Chat with comparative context from all uploaded resumes + job description."""
    context_sections = [
        f"--- Job Description ---\n{clean_text(job_description)[:3000]}"
    ]

    for i, file in enumerate(files, 1):
        resume_text = extract_text_from_pdf(file.file)
        resume_text = clean_text(resume_text)
        context_sections.append(
            f"--- Resume {i}: {file.filename} ---\n{resume_text[:2500]}"
        )
        file.file.seek(0)

    combined_context = "\n\n".join(context_sections)

    prompt = f"""You are an expert career advisor comparing multiple resumes against a job description.

User's Question: {query}

{combined_context}

IMPORTANT INSTRUCTIONS:
- Each resume above is labeled with a NUMBER, e.g. "Resume 1", "Resume 2", etc.
- First, identify the CANDIDATE'S NAME from each resume content (look for the name at the top of the resume).
- ALWAYS refer to candidates as "Resume 1 (Candidate Name)", e.g. "Resume 1 (John Doe)".
- NEVER mix up or swap which content belongs to which resume number.
- The resume NUMBER corresponds to the upload order chosen by the user.
- Answer based strictly on the resumes and job description above.
- Be specific about why one candidate is stronger or weaker than another.
- If asked for rankings, provide scores and clear justifications.
- Be concise and professional.
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Resume Enhancement Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/rewrite_resume")
async def rewrite_resume(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...)
):
    """Rewrite resume bullet points to better match a target job description."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an expert resume writer. Rewrite and enhance the resume below to better match the target job description.

Job Description:
{job_description[:3000]}

Current Resume:
{resume_text[:4000]}

Instructions:
- Rewrite bullet points to incorporate relevant keywords from the job description.
- Quantify achievements where possible (add realistic metrics if none exist).
- Use strong action verbs.
- Keep the same sections and structure but improve the content.
- Highlight transferable skills that align with the JD.
- Output the full enhanced resume in clean markdown format.
- Do NOT fabricate experience or skills not implied by the original resume.
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


@app.post("/keyword_optimizer")
async def keyword_optimizer(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...)
):
    """Analyze keyword gaps between resume and job description."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an ATS keyword optimization expert. Analyze the resume against the job description and identify keyword gaps.

Job Description:
{job_description[:3000]}

Resume:
{resume_text[:4000]}

Provide your analysis using these exact section headings (## markdown):

## ✅ Keywords Found in Resume
List keywords from the JD that already appear in the resume.

## ❌ Missing Keywords
List important keywords from the JD that are NOT in the resume.

## 📝 Where to Add Missing Keywords
For each missing keyword, suggest exactly WHERE in the resume it should be added and HOW to naturally incorporate it. Be specific about which section (Skills, Experience, Projects, etc.) and provide example bullet points.

## 📊 Keyword Match Score
Give an overall keyword match percentage (0-100%) and brief justification.

Rules:
- Focus on hard skills, technical terms, tools, and industry-specific language.
- Be specific and actionable.
- Use bullet points within each section.
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


@app.post("/cover_letter")
async def cover_letter(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form("the company"),
    tone: str = Form("professional")
):
    """Generate a tailored cover letter based on resume and job description."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""You are an expert cover letter writer. Write a compelling cover letter for a candidate applying to {company_name}.

Job Description:
{job_description[:3000]}

Candidate's Resume:
{resume_text[:4000]}

Instructions:
- Tone: {tone}
- Write a 3-4 paragraph cover letter.
- Opening: Hook the reader with enthusiasm for the specific role and company.
- Body: Highlight 2-3 most relevant experiences/skills from the resume that match the JD. Use specific examples.
- Closing: Express eagerness, mention availability, and include a call to action.
- Use the candidate's actual name if found in the resume.
- Do NOT use generic filler — every sentence should add value.
- Keep it under 400 words.
- Format as a professional letter (no markdown headings).
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


@app.post("/resume_summary")
async def resume_summary(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_description: str = Form(""),
    summary_type: str = Form("professional_summary")
):
    """Generate a professional summary, career objective, or LinkedIn headline."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    type_instructions = {
        "professional_summary": "Write 3 different professional summary options (3-4 sentences each). Each should highlight the candidate's key strengths, years of experience, technical skills, and value proposition.",
        "objective": "Write 3 different career objective options (2-3 sentences each). Each should state the candidate's career goals and what they bring to a prospective employer.",
        "headline": "Write 5 different LinkedIn headline options (under 120 characters each). Each should be punchy, keyword-rich, and highlight the candidate's expertise.",
    }

    instruction = type_instructions.get(summary_type, type_instructions["professional_summary"])

    jd_context = ""
    if job_description.strip():
        jd_context = f"""
Target Job Description (tailor the summary to this role):
{job_description[:2000]}
"""

    prompt = f"""You are an expert resume and personal branding consultant.

Candidate's Resume:
{resume_text[:4000]}
{jd_context}

Task: {instruction}

Rules:
- Base everything strictly on the resume content.
- Number each option (1, 2, 3, etc.).
- If a JD is provided, tailor the language to match the target role.
- Use strong, confident language — no generic fluff.
- Include relevant technical skills and domain expertise.
"""

    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}


# ═══════════════════════════════════════════════════════════════════════
# Resume Insights Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/skill_gap_analysis")
async def skill_gap_analysis(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """Analyze skill gaps by comparing resume skills against market-demanded skills from the vector store."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    # Extract candidate skills via LLM
    skill_prompt = f"""Extract ALL technical and professional skills from this resume. Return them as a simple comma-separated list, nothing else.

Resume:
{resume_text[:4000]}
"""
    skills_raw = get_llm_response(api_key, skill_prompt)
    candidate_skills = [s.strip().lower() for s in skills_raw.replace("\n", ",").split(",") if s.strip() and len(s.strip()) < 50]

    # Search vector store for matching job descriptions
    resume_chunks = splitter.split_text(resume_text)
    if resume_chunks:
        chunk_embeddings = [embeddings.embed_query(chunk) for chunk in resume_chunks]
        resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    else:
        resume_embedding = embeddings.embed_query(resume_text[:2000])

    job_results = job_vectorstore.similarity_search_by_vector(resume_embedding, k=10)
    job_context = "\n".join([doc.page_content for doc in job_results])

    # Extract market-demanded skills via LLM
    market_prompt = f"""Extract ALL required technical and professional skills from these job descriptions. Return them as a simple comma-separated list, nothing else.

Job Descriptions:
{job_context[:5000]}
"""
    market_raw = get_llm_response(api_key, market_prompt)
    market_skills = [s.strip().lower() for s in market_raw.replace("\n", ",").split(",") if s.strip() and len(s.strip()) < 50]

    # Deduplicate
    candidate_set = set(candidate_skills)
    market_set = set(market_skills)

    matched = sorted(candidate_set & market_set)
    missing = sorted(market_set - candidate_set)
    extra = sorted(candidate_set - market_set)

    match_pct = round(len(matched) / max(len(market_set), 1) * 100)

    return {
        "candidate_skills": sorted(candidate_set),
        "market_skills": sorted(market_set),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_percentage": match_pct,
    }


@app.post("/ats_score")
async def ats_score(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """Score resume for ATS compatibility across multiple categories."""
    resume_text = extract_text_from_pdf(file.file)
    # Light cleaning only — preserve case, emails, phone numbers, URLs, special chars
    resume_text = re.sub(r"\t", " ", resume_text)
    resume_text = re.sub(r"(\n|\r)+", "\n", resume_text)
    resume_text = re.sub(r" +", " ", resume_text).strip()

    # Pre-detect contact info with regex for accurate scoring
    has_email = bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", resume_text))
    has_phone = bool(re.search(r"(\+?\d[\d\s\-().]{7,}\d)", resume_text))
    has_linkedin = bool(re.search(r"linkedin", resume_text, re.IGNORECASE))
    has_github = bool(re.search(r"github", resume_text, re.IGNORECASE))

    contact_items = []
    contact_items.append(f"Email: {'FOUND' if has_email else 'NOT FOUND'}")
    contact_items.append(f"Phone: {'FOUND' if has_phone else 'NOT FOUND'}")
    contact_items.append(f"LinkedIn: {'FOUND' if has_linkedin else 'NOT FOUND'}")
    contact_items.append(f"GitHub: {'FOUND' if has_github else 'NOT FOUND'}")
    contact_summary = " | ".join(contact_items)

    # Calculate contact info score deterministically (not LLM-dependent)
    contact_score = 0
    if has_email: contact_score += 6
    if has_phone: contact_score += 6
    if has_linkedin: contact_score += 4
    if has_github: contact_score += 4
    contact_details_parts = []
    if has_email: contact_details_parts.append("Email found")
    else: contact_details_parts.append("Email missing")
    if has_phone: contact_details_parts.append("Phone found")
    else: contact_details_parts.append("Phone missing")
    if has_linkedin: contact_details_parts.append("LinkedIn found")
    else: contact_details_parts.append("LinkedIn missing")
    if has_github: contact_details_parts.append("GitHub found")
    else: contact_details_parts.append("GitHub missing")
    contact_details = ". ".join(contact_details_parts) + "."

    # Count words for resume length assessment
    word_count = len(resume_text.split())

    prompt = f"""You are an extremely strict ATS (Applicant Tracking System) auditor used by Fortune 500 companies. Your job is to find EVERY flaw. No resume is perfect — always find issues to deduct for.

VERIFIED CONTACT INFO: {contact_summary}
WORD COUNT: {word_count} words

Resume:
{resume_text[:4000]}

Respond in EXACTLY this format (one per line):
SECTION_STRUCTURE_SCORE: <number out of 25>
SECTION_STRUCTURE_DETAILS: <list specific issues found>
RESUME_LENGTH_SCORE: <number out of 15>
RESUME_LENGTH_DETAILS: <list specific issues found>
IMPACT_METRICS_SCORE: <number out of 15>
IMPACT_METRICS_DETAILS: <list specific issues found>
KEYWORD_RELEVANCE_SCORE: <number out of 15>
KEYWORD_RELEVANCE_DETAILS: <list specific issues found>
FORMATTING_READABILITY_SCORE: <number out of 10>
FORMATTING_READABILITY_DETAILS: <list specific issues found>

STRICT SCORING RULES:
- Section Structure: NEVER give more than 22. A score above 20 is nearly impossible.
- Resume Length: NEVER give more than 13. Most resumes are either too long or too short.
- Impact & Metrics: NEVER give more than 12. Most candidates fail to quantify achievements properly.
- Keyword Relevance: NEVER give more than 13. Keywords without context should be penalized.
- Formatting & Readability: NEVER give more than 8. There are always formatting improvements to be made.
- Always cite at least 2-3 specific issues per category to justify deductions.

SECTION STRUCTURE (max 25):
Required sections: Contact Info, Professional Summary/Objective, Work Experience, Education, Skills. Deduct heavily for: missing sections, unconventional section names, poor ordering, missing dates/locations in experience/education, lack of detail. Every missing standard section = at least 5 point deduction.

RESUME LENGTH (max 15):
Ideal: 300-800 words for entry-level, 500-1000 for mid-level. This resume has {word_count} words. Deduct if too short (lacks substance) or too verbose (wastes recruiter time). Only a perfectly sized resume scores above 12.

IMPACT & METRICS (max 15):
Be VERY strict here. Every bullet point should have quantified results (numbers, percentages, dollar amounts, user counts). Deduct for: vague descriptions ("improved performance"), missing metrics, generic achievements, lack of measurable results. Most resumes score 7-10 here.

KEYWORD RELEVANCE (max 15):
Check for: technical skills demonstrated in context (not just listed), industry-specific terminology, strong action verbs, relevant certifications. Deduct for: skills listed without usage context, missing industry buzzwords, weak or repetitive action verbs, generic soft skills.

FORMATTING & READABILITY (max 10):
Deduct for: long paragraphs instead of bullets, passive voice, inconsistent tense, buzzword stuffing, poor grammar, ATS-breaking elements (tables, columns, graphics, text boxes), inconsistent formatting.
"""

    feedback = get_llm_response(api_key, prompt)

    # Parse scores
    breakdown = {}

    # Contact Info is scored deterministically (not by LLM)
    breakdown["Contact Info"] = {"score": contact_score, "max": 20, "details": contact_details}

    # Section Structure is scored deterministically (not by LLM)
    resume_lower = resume_text.lower()
    required_sections = {
        "Education": bool(re.search(r"\beducation\b", resume_lower)),
        "Experience": bool(re.search(r"\b(experience|work\s*experience|professional\s*experience|internship)\b", resume_lower)),
        "Skills": bool(re.search(r"\b(skills|technical\s*skills|core\s*competencies)\b", resume_lower)),
        "Projects": bool(re.search(r"\b(projects|personal\s*projects|academic\s*projects)\b", resume_lower)),
        "Summary": bool(re.search(r"\b(summary|objective|profile|about\s*me)\b", resume_lower)),
    }
    section_score = 25
    section_details_parts = []
    for section_name, found in required_sections.items():
        if found:
            section_details_parts.append(f"{section_name} found")
        else:
            section_score -= 5
            section_details_parts.append(f"{section_name} missing")
    section_details = ". ".join(section_details_parts) + "."
    breakdown["Section Structure"] = {"score": section_score, "max": 25, "details": section_details}

    # LLM-scored categories
    categories = {
        "Resume Length": ("RESUME_LENGTH_SCORE", "RESUME_LENGTH_DETAILS", 15),
        "Impact & Metrics": ("IMPACT_METRICS_SCORE", "IMPACT_METRICS_DETAILS", 15),
        "Keyword Relevance": ("KEYWORD_RELEVANCE_SCORE", "KEYWORD_RELEVANCE_DETAILS", 15),
        "Formatting & Readability": ("FORMATTING_READABILITY_SCORE", "FORMATTING_READABILITY_DETAILS", 10),
    }

    total = contact_score + section_score
    for cat_name, (score_key, detail_key, max_score) in categories.items():
        score_match = re.search(rf"{score_key}[:\s]*(\d+)", feedback, re.IGNORECASE)
        detail_match = re.search(rf"{detail_key}[:\s]*(.*)", feedback, re.IGNORECASE)

        # Flexible fallback for Formatting & Readability (LLM often outputs different key names)
        if not score_match and "FORMATTING" in score_key:
            for pattern in [
                r"FORMATTING[\s_&AND]*READABILITY[\s_]*SCORE[:\s]*(\d+)",
                r"FORMATTING[:\s]*(\d+)",
                r"READABILITY[:\s]*(\d+)",
            ]:
                score_match = re.search(pattern, feedback, re.IGNORECASE)
                if score_match:
                    break
        if not detail_match and "FORMATTING" in detail_key:
            for pattern in [
                r"FORMATTING[\s_&AND]*READABILITY[\s_]*DETAILS[:\s]*(.*)",
                r"FORMATTING_DETAILS[:\s]*(.*)",
            ]:
                detail_match = re.search(pattern, feedback, re.IGNORECASE)
                if detail_match:
                    break

        score = min(int(score_match.group(1)), max_score) if score_match else 0
        details = detail_match.group(1).strip() if detail_match else "No details available."
        total += score
        breakdown[cat_name] = {"score": score, "max": max_score, "details": details}

    # Hard cap total at 95% — a perfect score should never be possible
    total = min(total, 95)

    return {
        "percentage": total,
        "breakdown": breakdown,
    }


# ═══════════════════════════════════════════════════════════════════════
# Job Search Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/job_recommendations")
async def job_recommendations(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    top_n: str = Form("5")
):
    """Find matching jobs from the vector store based on resume content."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    n = int(top_n)

    # Embed resume to find matching jobs
    resume_chunks = splitter.split_text(resume_text)
    if resume_chunks:
        chunk_embeddings = [embeddings.embed_query(chunk) for chunk in resume_chunks]
        resume_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    else:
        resume_embedding = embeddings.embed_query(resume_text[:2000])

    results = job_vectorstore.similarity_search_by_vector(resume_embedding, k=n)

    jobs = []
    for i, doc in enumerate(results):
        # Approximate similarity — best match first, linearly decay
        similarity = max(10, 100 - (i * int(70 / max(n, 1))))
        jobs.append({
            "content": doc.page_content[:2000],
            "similarity": similarity,
            "source": doc.metadata.get("source", "Unknown"),
        })

    # Sort by similarity descending
    jobs.sort(key=lambda x: x["similarity"], reverse=True)

    # Generate LLM summary analysis
    job_summaries = "\n\n".join([f"Job {i+1} (Match: {j['similarity']}%): {j['content'][:500]}" for i, j in enumerate(jobs[:5])])
    summary_prompt = f"""Briefly analyze how these top matching jobs align with the candidate's profile. 2-3 sentences max.

Candidate Resume:
{resume_text[:2000]}

Top Matching Jobs:
{job_summaries}
"""
    summary = get_llm_response(api_key, summary_prompt)

    return {
        "jobs": jobs,
        "total_found": len(jobs),
        "summary": summary,
    }


@app.post("/batch_jd_match")
async def batch_jd_match(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    job_descriptions: str = Form(...)
):
    """Score resume against multiple pasted job descriptions."""
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    jds = [jd.strip() for jd in job_descriptions.split("---JD---") if jd.strip()]
    results = []

    def parse_score(text, key):
        """Try multiple patterns to extract a score for a given key."""
        patterns = [
            rf"{key}[:\s]*(\d{{1,3}})",                # MATCH_SCORE: 75
            rf"{key}[:\s]*(\d{{1,3}})\s*/\s*\d+",      # MATCH_SCORE: 75/100
            rf"{key}[:\s]*(\d{{1,3}})%",                # MATCH_SCORE: 75%
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return min(int(m.group(1)), 100)
        return None

    for i, jd in enumerate(jds):
        prompt = f"""You are an expert resume evaluator. Score this resume against the job description.

Job Description:
{jd[:2000]}

Resume:
{resume_text[:3000]}

You MUST start your response with EXACTLY these three lines (each on its own line, numbers 0-100, no slashes, no percent signs):
MATCH_SCORE: <number>
SKILLS_FIT: <number>
EXPERIENCE_FIT: <number>

Then write a 1-2 sentence summary of the fit. No headings, no bullet points.
"""
        feedback = get_llm_response(api_key, prompt)

        score = parse_score(feedback, "MATCH_SCORE")
        skills_fit = parse_score(feedback, "SKILLS_FIT")
        experience_fit = parse_score(feedback, "EXPERIENCE_FIT")

        # If main score failed to parse, retry with a simpler prompt
        if score is None:
            retry_prompt = f"""Score this resume against the job description on a scale of 0 to 100.

Job Description (first 500 chars):
{jd[:500]}

Resume (first 500 chars):
{resume_text[:500]}

Reply with ONLY this format, nothing else:
MATCH_SCORE: <number>
SKILLS_FIT: <number>
EXPERIENCE_FIT: <number>
"""
            retry_feedback = get_llm_response(api_key, retry_prompt)
            score = parse_score(retry_feedback, "MATCH_SCORE")
            if skills_fit is None:
                skills_fit = parse_score(retry_feedback, "SKILLS_FIT")
            if experience_fit is None:
                experience_fit = parse_score(retry_feedback, "EXPERIENCE_FIT")

            # Use retry summary if original had no usable text
            if score is not None:
                feedback = retry_feedback

        # Final fallback — if still no score, try to find ANY number in the response
        if score is None:
            any_num = re.search(r"(\d{1,3})", feedback)
            score = min(int(any_num.group(1)), 100) if any_num else 30

        if skills_fit is None:
            skills_fit = score
        if experience_fit is None:
            experience_fit = score

        summary = re.sub(
            r"(?:MATCH_SCORE|SKILLS_FIT|EXPERIENCE_FIT)[:\s]*\d{1,3}[/\d%]*\s*",
            "", feedback
        ).strip()

        # Extract a title from the first meaningful line of the JD
        title_line = ""
        for line in jd.split("\n"):
            cleaned = line.strip()
            if cleaned and not re.match(r'^[\-=_*#~|/\\><: ]+$', cleaned):
                title_line = cleaned[:80]
                break
        if not title_line:
            title_line = f"Job Description {i+1}"

        results.append({
            "jd_index": i + 1,
            "jd_title": title_line or f"Job Description {i+1}",
            "match_score": score,
            "skills_fit": skills_fit,
            "experience_fit": experience_fit,
            "summary": summary,
            "jd_preview": jd[:300],
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
# --- IGNORE ---
# # Code to create and save vector stores (run once)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True)
# resume_docs = [Document(page_content=text, metadata={"source": file.filename}) for text in chunks]
# vectorstore.add_documents(resume_docs)
# vectorstore.save_local("vector_store/job_faiss")
