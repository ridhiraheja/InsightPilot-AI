# InsightPilot AI

**Turning Business Data into Explainable Decisions**

InsightPilot AI is a Streamlit app that takes a raw CSV or Excel file and turns it into something a non-technical person can act on. You upload a dataset, the app explores it for you, and then you can ask questions about it in plain English.

---

## Why we built it

Most businesses sit on a lot of data and still struggle to make decisions from it. Someone has to open the spreadsheet, work out the KPIs, spot the trends, and then explain what it all means. That takes time, and it's hard if you're not an analyst.

InsightPilot AI splits the job in two. Python and Pandas do the number crunching. A language model then explains the results in plain language. The model never gets to invent the numbers, because it only sees statistics we've already calculated from your file.

---

## What it does

**Data understanding.** On upload, the app works out the shape of your dataset: row and column counts, missing values, duplicate records, and which columns are numeric, categorical or dates.

**KPI analysis.** Mean, sum, minimum, maximum and standard deviation for the numeric columns.

**Dashboard.** Interactive Plotly charts for distributions, category comparisons, trends over time, and correlations.

**Trend detection.** If there's a date column, the app finds it and looks at how things change over time.

**Anomaly detection.** Isolation Forest flags unusual rows in the numeric data.

**Correlation analysis.** Finds relationships between numeric variables and points out the strongest ones.

**Natural language questions.** Ask things like:

- Which category has the highest average Sales?
- Which region generated the highest total Profit?
- What are the strongest correlations?
- What are the main data quality issues?
- Which product generated the highest total Sales?

**Explainable answers.** Every response comes back in the same five parts: Answer, Evidence, Business Interpretation, Recommendation, and Confidence. The evidence comes straight from your dataset.

**Download.** You can save the generated analysis and use it elsewhere.

---

## How it works

```text
Business Dataset
      |
      v
CSV / Excel Upload
      |
      v
Data Understanding
      |
      +-------------------+
      |                   |
      v                   v
  KPI Analysis       Data Quality
      |                   |
      +---------+---------+
                |
                v
      Deterministic Analysis
                |
       +--------+--------+
       |        |        |
       v        v        v
    Trends  Anomalies Correlations
       |        |        |
       +--------+--------+
                |
                v
        Verified Evidence
                |
                v
        AI Decision Engine
                |
                v
    Natural Language Question
                |
                v
  Answer + Evidence + Interpretation
      + Recommendation + Confidence
```

---

## Tech stack

| Area | Tools |
| --- | --- |
| Frontend | Streamlit |
| Data analysis | Python, Pandas, NumPy |
| Machine learning | Scikit-learn (Isolation Forest) |
| Visualization | Plotly |
| Generative AI | Google Gemini API, `google-genai` |
| File handling | CSV, Excel, OpenPyXL |

---

## Project structure

```text
InsightPilot-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

`secrets.toml` holds the Gemini API key, so it is not in the repository.

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/ridhiraheja/InsightPilot-AI.git
cd InsightPilot-AI
```

**2. Create a virtual environment**

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Gemini API setup

Create a file at `.streamlit/secrets.toml` and add your key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

Swap in your real key. And please don't commit this file to GitHub.

---

## Running the app

```bash
streamlit run app.py
```

It opens in your browser.

---

## Using it

1. Upload a CSV or Excel business dataset.
2. The app analyzes it automatically.
3. Look around: dataset overview, data quality, KPIs, distributions, category comparisons, trends, anomalies and correlations.
4. Type a business question, for example "Which category has the highest average Sales?"
5. Click **Analyze with InsightPilot AI**.
6. Read the Answer, Evidence, Business Interpretation, Recommendation and Confidence.

### Questions worth trying

- Which category has the highest average Sales?
- Which region generated the highest total Sales?
- Which region generated the highest total Profit?
- What are the strongest correlations in the dataset?
- What are the main data quality issues in this dataset?
- Which product generated the highest total Sales?
- Which category has the highest average Profit?

---

## Explainability and reliability

Language models are good at explaining things and unreliable at arithmetic. So we keep the two apart. Python and Pandas calculate the important statistics first, and only then does the AI model see them.

That setup helps in a few ways. There's less room for the model to make up numbers. Answers are tied to real evidence from the dataset. You can trace any figure back to where it came from. And the recommendations are built on verified statistics rather than guesses.

The model's job is to interpret results that have already been checked and explain them in business terms.

---

## Evaluation

We tested the system with business questions covering category, product and regional performance, profit, sales and quantity analysis, correlations, data quality, missing values, and recommendations.

What we looked at: numerical accuracy, answer correctness, how well answers were grounded in evidence, response time, and the share of queries that succeeded.

In early testing, the questions we tried all returned successful responses, usually within a few seconds depending on the AI request. This was a small informal test, not a formal benchmark.

---

## Target impact

The goal is to shorten the path from raw data to an explainable decision.

| Manual analysis | AI-assisted analysis |
| --- | --- |
| about 30 min | about 3 min |

That would be roughly a 90% cut in analysis time. To be clear, this is a target. It isn't a measured production result, and it shouldn't be read as one unless someone validates it independently.

---

## AI tools disclosure

We built this project with help from AI development tools. They were used for code generation and debugging, application structure, data analysis logic, UI work, documentation, and testing support.

The team reviewed and integrated the final application, architecture, testing and project decisions.

---

## Team

**DataForge AI**

- Ridhi Raheja, AI and Data Science Lead
- Shiven Pratap Singh, Full-Stack and Backend Developer

**Project:** InsightPilot AI
**Theme:** AI Decision Engine for Business Data
**Problem Statement:** PS-04

*Turning Business Data into Explainable Decisions*
