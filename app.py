import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import json
import re
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Analyzer Pro", page_icon="📄", layout="wide")


# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-match {
        background-color: #1b4332;
        color: #95d5b2;
        border: 1px solid #2d6a4f;
    }
    .skill-miss {
        background-color: #4a1525;
        color: #f4a0b5;
        border: 1px solid #7a2040;
    }
    .score-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .score-number {
        font-size: 3rem;
        font-weight: 700;
    }
    .score-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🔑 API Configuration")

api_key_input = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")
if api_key_input:
    st.session_state["api_key"] = api_key_input
api_key = st.session_state.get("api_key")
if api_key:
    st.sidebar.success("✅ API key is set!")

page = st.sidebar.radio(
    "Select Page",
    ["About", "Resume Details", "Resume Matching", "Chat with Resume and Job Description", "Compare Resumes", "Resume Insights", "Resume Enhancement", "Batch JD Matching"],
    index=0,
)

# ─── Shared Resume Upload (sidebar) ──────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Resume Upload")
sidebar_upload = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"], key="shared_resume")
if sidebar_upload:
    st.session_state["resume_bytes"] = sidebar_upload.getvalue()
    st.session_state["resume_name"] = sidebar_upload.name
    st.sidebar.success(f"✅ {sidebar_upload.name}")
elif st.session_state.get("resume_bytes"):
    st.sidebar.success(f"✅ {st.session_state['resume_name']} (persisted)")


class UploadedFileWrapper:
    """Makes stored bytes behave like an UploadedFile for call_backend."""
    def __init__(self, data, name):
        self._data = data
        self.name = name
    def getvalue(self):
        return self._data


def get_shared_resume():
    """Get the persisted resume from session state, or None."""
    data = st.session_state.get("resume_bytes")
    name = st.session_state.get("resume_name", "resume.pdf")
    return UploadedFileWrapper(data, name) if data else None


# ─── Helper ───────────────────────────────────────────────────────────

def call_backend(endpoint, file, data):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"http://localhost:8000/{endpoint}", files=files, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None


def call_backend_no_data(endpoint, file):
    """Call backend with only a file upload, no extra form data."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"http://localhost:8000/{endpoint}", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None


def save_jd_to_vectorstore(jd_text, button_key):
    """Show a button to save a job description to the vector store."""
    if jd_text and jd_text.strip():
        if st.button("💾 Save JD to Vector DB", key=button_key):
            with st.spinner("Saving job description to vector database..."):
                response = requests.post(
                    "http://localhost:8000/save_jd_to_vectorstore",
                    data={"job_description": jd_text},
                )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result.get('message', 'Failed to save.')}")
            else:
                st.error(f"Error: {response.text}")


def get_score_color(score):
    """Return a color based on score value."""
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#eab308"
    elif score >= 40:
        return "#f97316"
    else:
        return "#ef4444"


def create_gauge_chart(score, title="Overall Match Score"):
    """Create a gauge/donut chart for the overall score."""
    color = get_score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#1c1917"},
                {"range": [40, 60], "color": "#1c1917"},
                {"range": [60, 80], "color": "#1c1917"},
                {"range": [80, 100], "color": "#1c1917"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def create_radar_chart(scores_dict):
    """Create a radar chart for category scores."""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",
        line=dict(color="#818cf8", width=2),
        marker=dict(size=6, color="#818cf8"),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#334155", linecolor="#334155",
                tickfont=dict(color="#94a3b8", size=10),
            ),
            angularaxis=dict(
                gridcolor="#334155", linecolor="#334155",
                tickfont=dict(color="#e2e8f0", size=12),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def create_bar_chart(scores_dict):
    """Create a horizontal bar chart for category breakdown."""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    colors = [get_score_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 110], gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"), title="",
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=13), title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=10, r=30, t=10, b=10),
    )
    return fig


# ─── App Title ────────────────────────────────────────────────────────
st.title("📄 Resume Analyzer Pro")
st.markdown("Upload your resume to analyze, match, and chat — powered by AI & RAG.")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: Resume Details
# ═══════════════════════════════════════════════════════════════════════
if page == "Resume Details":
    st.header("📋 Resume Details Analysis")
    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to get started.")
        st.stop()

    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            data = {"api_key": api_key}
            result = call_backend("resume_details", uploaded_file, data)
        if result:
            st.session_state["resume_details_result"] = result

    if st.session_state.get("resume_details_result"):
        result = st.session_state["resume_details_result"]
        st.subheader("Extracted Resume Sections")
        feedback = result.get("llm_feedback", "")
        feedback = re.sub(r"\*\*(.+?)\*\*\s*:\s*", r"**\1** ", feedback)
        # Render with markdown so **bold** headings display prominently
        st.markdown("""
        <style>
            .resume-section strong {
                font-size: 1.25rem;
                color: #60a5fa;
                display: block;
                margin-top: 1rem;
                margin-bottom: 0.25rem;
                border-left: 3px solid #818cf8;
                padding-left: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="resume-section">\n\n{feedback}\n\n</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: Resume Matching (with charts)
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resume Matching":
    st.header("🎯 Resume-Job Matching")
    st.markdown("Paste a job description and see how well your resume matches.")

    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to continue.")
        st.stop()

    job_description = st.text_area(
        "📝 Job Description",
        value=st.session_state.get("job_description", ""),
        height=200,
        placeholder="Paste the job description here...",
    )
    st.session_state.job_description = job_description
    save_jd_to_vectorstore(job_description, "save_jd_matching")

    if not job_description.strip():
        st.info("✏️ Enter a job description to continue.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    if st.button("🚀 Match Resume to Job", type="primary"):
        with st.spinner("🔄 Analyzing match with AI..."):
            data = {"api_key": api_key, "job_description": job_description}
            result = call_backend("resume_matching", uploaded_file, data)

        if result and result.get("llm_feedback"):
            st.session_state["matching_result"] = result

    if st.session_state.get("matching_result"):
        result = st.session_state["matching_result"]
        feedback = result["llm_feedback"]

        # ── Extract scores from text using regex ──
        score_patterns = {
            "Total Match": r"TOTAL_MATCH_SCORE[:\s]*(\d{1,3})",
            "Skills": r"SKILLS_SCORE[:\s]*(\d{1,3})",
            "Experience": r"EXPERIENCE_SCORE[:\s]*(\d{1,3})",
            "Education": r"EDUCATION_SCORE[:\s]*(\d{1,3})",
            "Projects": r"PROJECTS_SCORE[:\s]*(\d{1,3})",
        }
        scores = {}
        for label, pattern in score_patterns.items():
            match = re.search(pattern, feedback, re.IGNORECASE)
            scores[label] = min(int(match.group(1)), 100) if match else 0

        overall = scores.get("Total Match", 0)

        # ── Remove score lines from feedback text for clean display ──
        clean_feedback = re.sub(
            r"(?:TOTAL_MATCH_SCORE|SKILLS_SCORE|EXPERIENCE_SCORE|EDUCATION_SCORE|PROJECTS_SCORE)[:\s]*\d{1,3}[/\d]*\s*",
            "", feedback
        ).strip()

        # ── Display gauge chart if we found an overall score ──
        if overall > 0:
            st.plotly_chart(create_gauge_chart(overall), use_container_width=True)

            # ── Category Score Cards ──
            category_scores = {k: v for k, v in scores.items() if k != "Total Match"}
            if any(v > 0 for v in category_scores.values()):
                st.markdown("#### 🏆 Category Scores")
                cols = st.columns(len(category_scores))
                for i, (cat, score) in enumerate(category_scores.items()):
                    with cols[i]:
                        color = get_score_color(score)
                        st.markdown(
                            f"""<div class="score-card">
                                <div class="score-number" style="color:{color}">{score}%</div>
                                <div class="score-label">{cat}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                # ── Charts ──
                col_radar, col_bar = st.columns(2)
                with col_radar:
                    st.subheader("📊 Radar Chart")
                    st.plotly_chart(create_radar_chart(category_scores), use_container_width=True)
                with col_bar:
                    st.subheader("📈 Category Breakdown")
                    st.plotly_chart(create_bar_chart(category_scores), use_container_width=True)

            st.markdown("---")

        
        st.subheader("📝 Detailed Feedback")
        st.markdown("""
        <style>
            .resume-section strong {
                font-size: 1.25rem;
                color: #60a5fa;
                display: block;
                margin-top: 1rem;
                margin-bottom: 0.25rem;
                border-left: 3px solid #818cf8;
                padding-left: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="resume-section">\n\n{clean_feedback}\n\n</div>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: Chat with Resume and Job Description
# ═══════════════════════════════════════════════════════════════════════
elif page == "Chat with Resume and Job Description":
    st.header("💬 Chat with Resume & Job Description")
    st.markdown("Ask questions about your resume, career advice, or interview prep — powered by RAG.")



    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    uploaded_file = get_shared_resume()

    # ── Save to Vector DB ──
    if uploaded_file:
        if st.button("💾 Save Resume to Vector DB"):
            with st.spinner("Saving to vector database..."):
                result = call_backend("save_resume_to_vectorstore", uploaded_file, {})
            if result and result.get("success"):
                st.success(f"✅ {result['message']}")
            elif result:
                st.error(f"❌ {result.get('message', 'Failed to save.')}")

    job_description = st.text_area(
        "📝 Job Description (optional)",
        value=st.session_state.get("job_description", ""),
        height=150,
        placeholder="Paste job description for context...",
    )
    st.session_state.job_description = job_description
    save_jd_to_vectorstore(job_description, "save_jd_chat")

    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to start chatting.")
        st.stop()

    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    st.markdown("---")

    # ── Chat History ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["response"])

    # ── Chat Input ──
    user_question = st.chat_input("Ask anything about your resume, skills, or career...")

    if user_question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data = {
                    "api_key": api_key,
                    "query": user_question,
                    "job_description": st.session_state.get("job_description", ""),
                }
                result = call_backend("chat_with_resume", uploaded_file, data)
                response = result.get("llm_feedback", "No response received.") if result else "No response received."
            st.write(response)

        # Store in history
        st.session_state.chat_history.append({
            "question": user_question,
            "response": response,
        })


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: Compare Resumes
# ═══════════════════════════════════════════════════════════════════════
elif page == "Compare Resumes":
    st.header("📊 Compare Multiple Resumes")
    st.markdown("Upload multiple resume PDFs and a job description to rank and compare candidates.")

    # ── Multi-file uploader ──
    uploaded_resumes = st.file_uploader(
        "📤 Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="compare_resumes_upload",
    )

    compare_jd = st.text_area(
        "📝 Job Description (required)",
        value=st.session_state.get("job_description", ""),
        height=200,
        placeholder="Paste the job description here...",
        key="compare_jd",
    )
    save_jd_to_vectorstore(compare_jd, "save_jd_compare")

    if not uploaded_resumes:
        st.info("📤 Upload 2 or more resume PDFs above to compare.")
        st.stop()
    if len(uploaded_resumes) < 2:
        st.info("📤 Please upload at least 2 resumes to compare.")
        st.stop()
    if not compare_jd.strip():
        st.info("✏️ Enter a job description to compare resumes against.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    st.success(f"✅ {len(uploaded_resumes)} resumes uploaded")

    # Show upload order for reference
    st.caption("**Upload Order:** " + ", ".join([f"Resume {i}: {f.name}" for i, f in enumerate(uploaded_resumes, 1)]))

    # ── Compare button ──
    if st.button("🚀 Compare Resumes", type="primary"):
        with st.spinner(f"🔄 Scoring {len(uploaded_resumes)} resumes against the job description..."):
            files = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_resumes
            ]
            data = {"api_key": api_key, "job_description": compare_jd}
            response = requests.post(
                "http://localhost:8000/compare_resumes",
                files=files,
                data=data,
            )
        if response.status_code == 200:
            st.session_state["compare_results"] = response.json().get("results", [])
        else:
            st.error(f"Error: {response.text}")

    # ── Display results ──
    if st.session_state.get("compare_results"):
        results = st.session_state["compare_results"]
        st.markdown("---")

        # ── Best match highlight ──
        best = results[0]
        best_color = get_score_color(best["total_score"])
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 2px solid {best_color}; border-radius: 12px; padding: 1.5rem;
                text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 0.9rem; color: #94a3b8;">🏆 Best Match</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #e2e8f0;">Resume {best.get('upload_order', '?')} — {best['filename']}</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {best_color};">{best['total_score']}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Ranking table ──
        st.subheader("🏅 Ranking")
        for rank, r in enumerate(results, 1):
            color = get_score_color(r["total_score"])
            medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"

            with st.container():
                st.markdown(
                    f"""<div style="background: #1e293b; border-left: 4px solid {color};
                        border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.3rem;">{medal}</span>
                                <span style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-left: 0.5rem;">Resume {r.get('upload_order', '?')} — {r['filename']}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{r['total_score']}%</div>
                        </div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">
                            <span>Skills: <b style="color:#e2e8f0">{r['skills']}%</b></span>
                            <span>Experience: <b style="color:#e2e8f0">{r['experience']}%</b></span>
                            <span>Education: <b style="color:#e2e8f0">{r['education']}%</b></span>
                            <span>Projects: <b style="color:#e2e8f0">{r['projects']}%</b></span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander(f"📝 Summary — Resume {r.get('upload_order', '?')}: {r['filename']}"):
                    st.write(r.get("summary", "No summary available."))

        # ── Save to Vector DB ──
        st.markdown("---")
        st.subheader("💾 Save Resumes to Vector DB")
        st.markdown("Select resumes to save to the vector store for future searches and recommendations.")

        selected_to_save = []
        for i, r in enumerate(results):
            if st.checkbox(f"📄 Resume {r.get('upload_order', '?')}: {r['filename']} (Score: {r['total_score']}%)", key=f"save_cb_{i}"):
                selected_to_save.append(r['filename'])

        if selected_to_save:
            if st.button(f"💾 Save {len(selected_to_save)} Resume(s) to Vector DB", key="save_to_vdb_btn"):
                saved = 0
                for f in uploaded_resumes:
                    if f.name in selected_to_save:
                        with st.spinner(f"Saving {f.name}..."):
                            files = {"file": (f.name, f.getvalue(), "application/pdf")}
                            resp = requests.post(
                                "http://localhost:8000/save_resume_to_vectorstore",
                                files=files,
                                data={},
                            )
                            if resp.status_code == 200 and resp.json().get("success"):
                                st.success(f"✅ {f.name} — {resp.json()['message']}")
                                saved += 1
                            else:
                                msg = resp.json().get("message", resp.text) if resp.status_code == 200 else resp.text
                                st.error(f"❌ {f.name} — {msg}")
                if saved:
                    st.balloons()

        # ── Comparative chat ──
        st.markdown("---")
        st.subheader("💬 Chat About the Comparison")

        if "compare_chat_history" not in st.session_state:
            st.session_state.compare_chat_history = []

        for chat in st.session_state.compare_chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["response"])

        compare_question = st.chat_input(
            "Ask about the comparison (e.g., 'Why is Resume 1 better than Resume 2?')...",
            key="compare_chat_input",
        )

        if compare_question:
            with st.chat_message("user"):
                st.write(compare_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    files = [
                        ("files", (f.name, f.getvalue(), "application/pdf"))
                        for f in uploaded_resumes
                    ]
                    data = {
                        "api_key": api_key,
                        "query": compare_question,
                        "job_description": compare_jd,
                    }
                    resp = requests.post(
                        "http://localhost:8000/chat_with_comparison",
                        files=files,
                        data=data,
                    )
                    if resp.status_code == 200:
                        answer = resp.json().get("llm_feedback", "No response.")
                    else:
                        answer = f"Error: {resp.text}"
                st.write(answer)

            st.session_state.compare_chat_history.append({
                "question": compare_question,
                "response": answer,
            })



# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: Resume Insights (Skill Gap + ATS Score)
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resume Insights":
    st.header("🔬 Resume Insights")
    st.markdown("Skill Gap Analysis & ATS Score — understand your resume's strengths and weaknesses.")

    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to get started.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    insight_tab1, insight_tab2 = st.tabs(["📊 Skill Gap Analysis", "🤖 ATS Score"])

    # ── TAB 1: Skill Gap Analysis ──
    with insight_tab1:
        st.subheader("📊 Skill Gap Analysis")
        st.markdown("Compare your skills against what the market demands for roles matching your profile.")

        if st.button("🔍 Analyze Skill Gap", key="skill_gap_btn"):
            with st.spinner("🔄 Extracting skills and searching market demands..."):
                data = {"api_key": api_key}
                result = call_backend("skill_gap_analysis", uploaded_file, data)

            if result and "candidate_skills" in result:
                st.session_state["skill_gap"] = result

        if st.session_state.get("skill_gap"):
            result = st.session_state["skill_gap"]
            matched = result["matched_skills"]
            missing = result["missing_skills"]
            extra = result["extra_skills"]
            pct = result["match_percentage"]

            # ── Match percentage gauge ──
            st.plotly_chart(create_gauge_chart(pct, "Market Skill Match"), use_container_width=True)

            # ── Summary metrics ──
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="score-card"><div class="score-number" style="color:#22c55e">{len(matched)}</div><div class="score-label">Matched Skills</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="score-card"><div class="score-number" style="color:#ef4444">{len(missing)}</div><div class="score-label">Missing Skills</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="score-card"><div class="score-number" style="color:#60a5fa">{len(extra)}</div><div class="score-label">Extra Skills</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Matched skills (green tags) ──
            if matched:
                st.markdown("#### ✅ Skills You Have That the Market Wants")
                tags = " ".join([f'<span class="skill-tag skill-match">{s.title()}</span>' for s in matched])
                st.markdown(tags, unsafe_allow_html=True)

            # ── Missing skills (red tags) ──
            if missing:
                st.markdown("#### ❌ Skills the Market Wants That You're Missing")
                tags = " ".join([f'<span class="skill-tag skill-miss">{s.title()}</span>' for s in missing])
                st.markdown(tags, unsafe_allow_html=True)

            # ── Extra skills (blue tags) ──
            if extra:
                st.markdown("#### 💡 Your Unique Skills (Not Commonly Demanded)")
                tags = " ".join([f'<span style="display:inline-block;padding:4px 12px;margin:3px;border-radius:16px;font-size:0.85rem;background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;">{s.title()}</span>' for s in extra])
                st.markdown(tags, unsafe_allow_html=True)

            # ── Bar chart ──
            if matched or missing:
                st.markdown("---")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[len(matched), len(missing), len(extra)],
                    y=["Matched", "Missing", "Extra"],
                    orientation="h",
                    marker=dict(color=["#22c55e", "#ef4444", "#60a5fa"]),
                    text=[len(matched), len(missing), len(extra)],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=14),
                ))
                fig.update_layout(
                    title="Skill Distribution",
                    xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
                    yaxis=dict(tickfont=dict(color="#e2e8f0", size=14)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=250, margin=dict(l=10, r=40, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: ATS Score ──
    with insight_tab2:
        st.subheader("🤖 ATS Compatibility Score")
        st.markdown("Check how well your resume is optimized for Applicant Tracking Systems.")

        if st.button("🔍 Check ATS Score", key="ats_btn"):
            with st.spinner("🔄 Analyzing ATS compatibility..."):
                result = call_backend("ats_score", uploaded_file, {"api_key": api_key})

            if result and "percentage" in result:
                st.session_state["ats_result"] = result

        if st.session_state.get("ats_result"):
            result = st.session_state["ats_result"]
            pct = result["percentage"]

            # ── Overall gauge ──
            st.plotly_chart(create_gauge_chart(pct, "ATS Compatibility Score"), use_container_width=True)

            # ── Category breakdown ──
            st.markdown("#### 📋 Category Breakdown")
            breakdown = result["breakdown"]

            for cat_name, cat_data in breakdown.items():
                score = cat_data["score"]
                max_score = cat_data["max"]
                details = cat_data["details"]
                pct_cat = round(score / max_score * 100) if max_score else 0
                color = get_score_color(pct_cat)

                st.markdown(
                    f"""<div style="background:#1e293b; border-radius:8px; padding:1rem; margin-bottom:0.75rem; border-left:4px solid {color};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; color:#e2e8f0;">{cat_name}</span>
                            <span style="font-weight:700; color:{color};">{score}/{max_score}</span>
                        </div>
                        <div style="background:#0f172a; border-radius:4px; height:8px; margin:0.5rem 0;">
                            <div style="background:{color}; width:{pct_cat}%; height:100%; border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8;">{details}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── Tips ──
            st.markdown("---")
            st.markdown("#### 💡 Quick Tips to Improve Your ATS Score")
            tips = []
            if breakdown.get("Contact Info", {}).get("score", 0) < 16:
                tips.append("Add missing contact info ")
            if breakdown.get("Section Structure", {}).get("score", 0) < 18:
                tips.append("Use standard section headings: Education, Experience, Skills, Projects")
            if breakdown.get("Resume Length", {}).get("score", 0) < 10:
                tips.append("Aim for 300-800 words — concise but comprehensive")
            if breakdown.get("Impact & Metrics", {}).get("score", 0) < 10:
                tips.append("Quantify achievements with numbers and use action verbs (led, built, improved)")
            if breakdown.get("Keyword Relevance", {}).get("score", 0) < 10:
                tips.append("Demonstrate skills in context, not just as a list — show how you used them")
            if breakdown.get("Formatting & Readability", {}).get("score", 0) < 7:
                tips.append("Use clean formatting with concise bullet points and consistent tense")
            if not tips:
                tips.append("Your resume looks well-optimized for ATS! 🎉")
            for tip in tips:
                st.write(f"• {tip}")



# ═══════════════════════════════════════════════════════════════════════
# PAGE 6: Resume Enhancement
# ═══════════════════════════════════════════════════════════════════════
elif page == "Resume Enhancement":
    st.header("✍️ Resume Enhancement")
    st.markdown("AI-powered tools to optimize your resume for any target role.")

    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to get started.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "🔄 AI Rewriter",
        "💌 Cover Letter", "📝 Summary Generator"
    ])

    # ── TAB 1: AI Resume Rewriter ──
    with tab1:
        st.subheader("🔄 AI Resume Rewriter")
        st.markdown("Rewrite your resume bullet points to better match a target job description.")

        rewrite_jd = st.text_area(
            "📋 Target Job Description",
            height=180,
            placeholder="Paste the job description you want to tailor your resume for...",
            key="rewrite_jd",
        )
        save_jd_to_vectorstore(rewrite_jd, "save_jd_rewriter")

        if not rewrite_jd.strip():
            st.info("✏️ Paste a job description above to get rewrite suggestions.")
        elif st.button("🚀 Rewrite Resume", key="rewrite_btn", type="primary"):
            with st.spinner("🔄 Rewriting bullet points to match the JD..."):
                result = call_backend("rewrite_resume", uploaded_file, {
                    "api_key": api_key,
                    "job_description": rewrite_jd,
                })
            if result and "llm_feedback" in result:
                st.session_state["rewrite_result"] = result["llm_feedback"]

        if st.session_state.get("rewrite_result"):
            st.markdown("---")
            st.markdown("#### ✅ Enhanced Resume")
            st.markdown(st.session_state["rewrite_result"])


    # ── TAB 2: Cover Letter Generator ──
    with tab2:
        st.subheader("💌 Cover Letter Generator")
        st.markdown("Generate a tailored cover letter based on your resume and the target job.")

        cl_col1, cl_col2 = st.columns(2)
        with cl_col1:
            company_name = st.text_input("🏢 Company Name", placeholder="e.g., Google", key="cl_company")
        with cl_col2:
            tone = st.selectbox("🎨 Tone", ["Professional", "Enthusiastic", "Conversational", "Formal"], key="cl_tone")

        cover_jd = st.text_area(
            "📋 Job Description",
            height=180,
            placeholder="Paste the job description for the cover letter...",
            key="cover_jd",
        )
        save_jd_to_vectorstore(cover_jd, "save_jd_cover")

        if not cover_jd.strip():
            st.info("✏️ Paste a job description above to generate a cover letter.")
        elif st.button("📝 Generate Cover Letter", key="cl_btn", type="primary"):
            with st.spinner("🔄 Crafting your personalized cover letter..."):
                result = call_backend("cover_letter", uploaded_file, {
                    "api_key": api_key,
                    "job_description": cover_jd,
                    "company_name": company_name or "the company",
                    "tone": tone.lower(),
                })
            if result and "llm_feedback" in result:
                st.session_state["cover_result"] = result["llm_feedback"]

        if st.session_state.get("cover_result"):
            st.markdown("---")
            st.markdown("#### 💌 Your Cover Letter")
            st.markdown(
                f'<div style="background:#1e293b; border-radius:12px; padding:1.5rem; '
                f'border-left:4px solid #6366f1; line-height:1.8; color:#e2e8f0;">'
                f'{st.session_state["cover_result"]}</div>',
                unsafe_allow_html=True,
            )

    # ── TAB 3: Resume Summary Generator ──
    with tab3:
        st.subheader("📝 Summary / Objective Generator")
        st.markdown("Generate a professional summary, career objective, or headline tailored to a role.")

        summary_type = st.selectbox(
            "📌 Summary Type",
            ["Professional Summary", "Career Objective", "LinkedIn Headline"],
            key="summary_type",
        )
        type_map = {
            "Professional Summary": "professional_summary",
            "Career Objective": "objective",
            "LinkedIn Headline": "headline",
        }

        summary_jd = st.text_area(
            "📋 Target Job Description (optional — for role-specific tailoring)",
            height=150,
            placeholder="Paste a JD to tailor the summary to a specific role, or leave blank for a general summary...",
            key="summary_jd",
        )
        save_jd_to_vectorstore(summary_jd, "save_jd_summary")

        if st.button("✨ Generate Summary", key="summary_btn", type="primary"):
            with st.spinner("🔄 Generating tailored summary options..."):
                result = call_backend("resume_summary", uploaded_file, {
                    "api_key": api_key,
                    "job_description": summary_jd,
                    "summary_type": type_map[summary_type],
                })
            if result and "llm_feedback" in result:
                st.session_state["summary_result"] = result["llm_feedback"]

        if st.session_state.get("summary_result"):
            st.markdown("---")
            st.markdown("#### ✨ Generated Options")
            st.markdown(st.session_state["summary_result"])



# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# PAGE 7: Batch JD Matching
# ═══════════════════════════════════════════════════════════════════════
elif page == "Batch JD Matching":
    st.header("📋 Batch JD Matching")
    st.markdown("Paste multiple job descriptions and rank them by fit against your resume.")

    uploaded_file = get_shared_resume()
    if not uploaded_file:
        st.info("📤 Please upload a resume PDF in the sidebar to get started.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()
    st.info("Separate each job description with `---JD---` on its own line.")

    batch_jds = st.text_area(
        "📋 Paste Multiple Job Descriptions",
        height=300,
        placeholder="Software Engineer at Google\nWe are looking for...\n\n---JD---\n\nData Scientist at Amazon\nWe need a candidate who...\n\n---JD---\n\nML Engineer at Meta\nExciting opportunity for...",
        key="batch_jds",
    )

    # Save all batch JDs to vector store
    if batch_jds.strip():
        parsed_jds = [j.strip() for j in batch_jds.split("---JD---") if j.strip()]
        if parsed_jds and st.button(f"💾 Save {len(parsed_jds)} JD(s) to Vector DB", key="save_jd_batch"):
            saved = 0
            for idx, jd in enumerate(parsed_jds, 1):
                with st.spinner(f"Saving JD {idx}/{len(parsed_jds)}..."):
                    resp = requests.post(
                        "http://localhost:8000/save_jd_to_vectorstore",
                        data={"job_description": jd},
                    )
                if resp.status_code == 200 and resp.json().get("success"):
                    st.success(f"✅ JD {idx} — {resp.json()['message']}")
                    saved += 1
                else:
                    msg = resp.json().get("message", resp.text) if resp.status_code == 200 else resp.text
                    st.error(f"❌ JD {idx} — {msg}")
            if saved:
                st.balloons()

    if not batch_jds.strip():
        st.info("✏️ Paste job descriptions above, separated by `---JD---`.")
    else:
        jd_count = len([j for j in batch_jds.split("---JD---") if j.strip()])
        st.success(f"✅ {jd_count} job description(s) detected")

        if st.button(f"🚀 Match Against {jd_count} JDs", key="batch_btn", type="primary"):
            with st.spinner(f"🔄 Scoring resume against {jd_count} job descriptions..."):
                result = call_backend("batch_jd_match", uploaded_file, {
                    "api_key": api_key,
                    "job_descriptions": batch_jds,
                })
            if result and "results" in result:
                st.session_state["batch_result"] = result["results"]

    if st.session_state.get("batch_result"):
        results = st.session_state["batch_result"]
        st.markdown("---")

        # Best match highlight
        best = results[0]
        best_color = get_score_color(best["match_score"])
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 2px solid {best_color}; border-radius: 12px; padding: 1.5rem;
                text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 0.9rem; color: #94a3b8;">🏆 Best Matching Job</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #e2e8f0;">{best['jd_title']}</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {best_color};">{best['match_score']}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Ranked list
        st.subheader("🏅 Rankings")
        for rank, r in enumerate(results, 1):
            color = get_score_color(r["match_score"])
            medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"

            st.markdown(
                f"""<div style="background:#1e293b; border-left:4px solid {color};
                    border-radius:8px; padding:1rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:1.3rem;">{medal}</span>
                            <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-left:0.5rem;">{r['jd_title']}</span>
                        </div>
                        <div style="font-size:1.5rem; font-weight:700; color:{color};">{r['match_score']}%</div>
                    </div>
                    <div style="display:flex; gap:1.5rem; margin-top:0.5rem; font-size:0.85rem; color:#94a3b8;">
                        <span>Skills Fit: <b style="color:#e2e8f0">{r['skills_fit']}%</b></span>
                        <span>Experience Fit: <b style="color:#e2e8f0">{r['experience_fit']}%</b></span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander(f"📝 Summary — {r['jd_title']}"):
                st.write(r.get("summary", "No summary available."))



# ═══════════════════════════════════════════════════════════════════════
# PAGE 8: About
# ═══════════════════════════════════════════════════════════════════════
elif page == "About":
    st.header("ℹ️ About Resume Analyzer Pro")
    st.markdown("""
    ### Welcome to Resume Analyzer Pro!

    This AI-powered application helps you analyze your resume against job descriptions and provides personalized recommendations to improve your career prospects.

    ### Features:
    - **Intelligent Resume Processing**: Upload PDF resumes with section-wise parsing and content visualization
    - **LLM-Powered Extraction**: AI-based extraction of skills, education, projects, work experience, and certifications via the Hugging Face Inference API
    - **Smart Job Matching**: Semantic matching using sentence-transformer embeddings and FAISS, with visual score charts
    - **Enhanced Chat Interface**: Context-aware conversations powered by a Retrieval-Augmented Generation (RAG) flow
    - **Vector Store Integration**: Save your resume into the FAISS vector database for enhanced retrieval
    - **Flexible Chat Sources**: Choose to chat with your resume, job descriptions, or the full vector store

    ### Technology Stack:
    - **RAG Pipeline**: Retrieval-Augmented Generation using LangChain text splitting and context assembly
    - **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` for semantic embeddings
    - **FAISS**: Local vector stores for efficient similarity search (`vector_store/`)
    - **LLM Inference**: Hugging Face Inference API with `mistralai/Mistral-7B-Instruct-v0.2`
    - **FastAPI**: Backend service exposing analysis, matching, chat, and vector store endpoints
    - **Streamlit**: Interactive web interface with Plotly charts
    - **PyPDF2**: PDF text extraction

    ### How It Works:
    1. **Upload & Process**: Upload your PDF resume; text is extracted locally with PyPDF2 and cleaned
    2. **Embedding & Retrieval**: Resume/job text is split into chunks, embedded with a sentence transformer, and searched in local FAISS stores
    3. **Smart Matching**: The LLM evaluates resume vs. job description with retrieved context to generate structured scores and feedback
    4. **Chat**: Ask questions; the system augments your query with relevant retrieved context and responds via the LLM
    5. **Save to Vector Store**: Store your resume in the FAISS database for future retrieval-augmented conversations

    ### Privacy Notice:
    Your resume data is processed locally and stored only for your session. We do not share your personal information with third parties.

    ### Contact:
    For support or feedback, please contact: @garvity
    """)

    st.markdown("---")
    st.markdown("Resume Analyzer Pro © 2026 | Powered by AI and Machine Learning")
