from scripts.powerExtract import ResumeJobMatchingSystem, analyze_job_fit_hard_skills, analyze_job_fit_soft_skills
import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text
    
pdf_file = 'data/Resumes/0ecaf8f504204421b4a059d1c50d2895_resume.pdf'
text = read_pdf(pdf_file)

job_description = """Responsibilities
Help determine which methods are appropriate to explore, mine, analyze, and manipulate massive data sets.
Develop and analyze modeling techniques using creativity, mathematics, machine learning, and statistical methods.
Create user-friendly reports using databases, analyses, programs, and scripts.
Ensure connectivity between databases and systems by working with other teams.
Collaborate with stakeholders to develop solutions to analytical problems.
Communicate results in a creative and understandable manner through storytelling.

Skills
check
Represents the skills you have
checkMachine Learning
Deep Learning
checkNLP
Computer Vision
checkPredictive Modeling
checkStatistics
Software Delivery
checkData Visualization
TS/SCI clearance
Significant coursework or professional experience in machine learning, deep learning, NLP, computer vision, predictive modeling and forecasting, statistics, or similar analytic domain
Experience delivering software or technical solutions to DOD or IC customers, USAF, JAIC, NRO, NGA, or other intelligence organizations strongly desired
Familiarity with data visualization software, tools, and techniques is desired"""

json_file_path = 'data/job_skills.json'
csv_file_path = 'data/job_skills.csv'
ss_json_file_path = 'data/soft_skills.json'
matching_system = ResumeJobMatchingSystem(json_file_path, csv_file_path, ss_json_file_path)
if matching_system.df is not None and matching_system.soft_skills_df is not None:
    resume_keywords, job_keywords = analyze_job_fit_hard_skills(matching_system)
    resume_soft_keywords, job_soft_keywords = analyze_job_fit_soft_skills(matching_system)

print(resume_keywords)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(job_keywords)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(resume_soft_keywords)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(job_soft_keywords)


