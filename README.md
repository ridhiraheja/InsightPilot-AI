\# InsightPilot AI

&#x20;

\*\*Turning Business Data into Explainable Decisions\*\*

&#x20;

InsightPilot AI is a Streamlit app that takes a raw CSV or Excel file and turns it into something a non-technical person can act on. You upload a dataset, the app explores it for you, and then you can ask questions about it in plain English.

&#x20;

\---

&#x20;

\## Why we built it

&#x20;

Most businesses sit on a lot of data and still struggle to make decisions from it. Someone has to open the spreadsheet, work out the KPIs, spot the trends, and then explain what it all means. That takes time, and it's hard if you're not an analyst.

&#x20;

InsightPilot AI splits the job in two. Python and Pandas do the number crunching. A language model then explains the results in plain language. The model never gets to invent the numbers, because it only sees statistics we've already calculated from your file.

&#x20;

\---

&#x20;

\## What it does

&#x20;

\*\*Data understanding.\*\* On upload, the app works out the shape of your dataset: row and column counts, missing values, duplicate records, and which columns are numeric, categorical or dates.

&#x20;

\*\*KPI analysis.\*\* Mean, sum, minimum, maximum and standard deviation for the numeric columns.

&#x20;

\*\*Dashboard.\*\* Interactive Plotly charts for distributions, category comparisons, trends over time, and correlations.

&#x20;

\*\*Trend detection.\*\* If there's a date column, the app finds it and looks at how things change over time.

&#x20;

\*\*Anomaly detection.\*\* Isolation Forest flags unusual rows in the numeric data.

&#x20;

\*\*Correlation analysis.\*\* Finds relationships between numeric variables and points out the strongest ones.

&#x20;

\*\*Natural language questions.\*\* Ask things like:

&#x20;

\- Which category has the highest average Sales?

\- Which region generated the highest total Profit?

\- What are the strongest correlations?

\- What are the main data quality issues?

\- Which product generated the highest total Sales?

\*\*Explainable answers.\*\* Every response comes back in the same five parts: Answer, Evidence, Business Interpretation, Recommendation, and Confidence. The evidence comes straight from your dataset.

&#x20;

\*\*Download.\*\* You can save the generated analysis and use it elsewhere.

&#x20;

\---

&#x20;

\## How it works

&#x20;

```text

Business Dataset

&#x20;     |

&#x20;     v

CSV / Excel Upload

&#x20;     |

&#x20;     v

Data Understanding

&#x20;     |

&#x20;     +-------------------+

&#x20;     |                   |

&#x20;     v                   v

&#x20; KPI Analysis       Data Quality

&#x20;     |                   |

&#x20;     +---------+---------+

&#x20;               |

&#x20;               v

&#x20;     Deterministic Analysis

&#x20;               |

&#x20;      +--------+--------+

&#x20;      |        |        |

&#x20;      v        v        v

&#x20;   Trends  Anomalies Correlations

&#x20;      |        |        |

&#x20;      +--------+--------+

&#x20;               |

&#x20;               v

&#x20;       Verified Evidence

&#x20;               |

&#x20;               v

&#x20;       AI Decision Engine

&#x20;               |

&#x20;               v

&#x20;   Natural Language Question

&#x20;               |

&#x20;               v

&#x20; Answer + Evidence + Interpretation

&#x20;     + Recommendation + Confidence

```

&#x20;

\---

&#x20;

\## Tech stack

&#x20;

| Area | Tools |

| --- | --- |

| Frontend | Streamlit |

| Data analysis | Python, Pandas, NumPy |

| Machine learning | Scikit-learn (Isolation Forest) |

| Visualization | Plotly |

| Generative AI | Google Gemini API, `google-genai` |

| File handling | CSV, Excel, OpenPyXL |

&#x20;

\---

&#x20;

\## Project structure

&#x20;

```text

InsightPilot-AI/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

└── .streamlit/

&#x20;   └── secrets.toml

```

&#x20;

`secrets.toml` holds the Gemini API key, so it is not in the repository.

&#x20;

\---

&#x20;

\## Installation

&#x20;

\*\*1. Clone the repo\*\*

&#x20;

```bash

git clone https://github.com/ridhiraheja/InsightPilot-AI.git

cd InsightPilot-AI

```

&#x20;

\*\*2. Create a virtual environment\*\*

&#x20;

On Windows:

&#x20;

```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```

&#x20;

\*\*3. Install dependencies\*\*

&#x20;

```bash

pip install -r requirements.txt

```

&#x20;

\---

&#x20;

\## Gemini API setup

&#x20;

Create a file at `.streamlit/secrets.toml` and add your key:

&#x20;

```toml

GEMINI\_API\_KEY = "YOUR\_API\_KEY\_HERE"

```

&#x20;

Swap in your real key. And please don't commit this file to GitHub.

&#x20;

\---

&#x20;

\## Running the app

&#x20;

```bash

streamlit run app.py

```

&#x20;

It opens in your browser.

&#x20;

\---

&#x20;

\## Using it

&#x20;

1\. Upload a CSV or Excel business dataset.

2\. The app analyzes it automatically.

3\. Look around: dataset overview, data quality, KPIs, distributions, category comparisons, trends, anomalies and correlations.

4\. Type a business question, for example "Which category has the highest average Sales?"

5\. Click \*\*Analyze with InsightPilot AI\*\*.

6\. Read the Answer, Evidence, Business Interpretation, Recommendation and Confidence.

\### Questions worth trying

&#x20;

\- Which category has the highest average Sales?

\- Which region generated the highest total Sales?

\- Which region generated the highest total Profit?

\- What are the strongest correlations in the dataset?

\- What are the main data quality issues in this dataset?

\- Which product generated the highest total Sales?

\- Which category has the highest average Profit?

\---

&#x20;

\## Explainability and reliability

&#x20;

Language models are good at explaining things and unreliable at arithmetic. So we keep the two apart. Python and Pandas calculate the important statistics first, and only then does the AI model see them.

&#x20;

That setup helps in a few ways. There's less room for the model to make up numbers. Answers are tied to real evidence from the dataset. You can trace any figure back to where it came from. And the recommendations are built on verified statistics rather than guesses.

&#x20;

The model's job is to interpret results that have already been checked and explain them in business terms.

&#x20;

\---

&#x20;

\## Evaluation

&#x20;

We tested the system with business questions covering category, product and regional performance, profit, sales and quantity analysis, correlations, data quality, missing values, and recommendations.

&#x20;

What we looked at: numerical accuracy, answer correctness, how well answers were grounded in evidence, response time, and the share of queries that succeeded.

&#x20;

In early testing, the questions we tried all returned successful responses, usually within a few seconds depending on the AI request. This was a small informal test, not a formal benchmark.

&#x20;

\---

&#x20;

\## Target impact

&#x20;

The goal is to shorten the path from raw data to an explainable decision.

&#x20;

| Manual analysis | AI-assisted analysis |

| --- | --- |

| about 30 min | about 3 min |

&#x20;

That would be roughly a 90% cut in analysis time. To be clear, this is a target. It isn't a measured production result, and it shouldn't be read as one unless someone validates it independently.

&#x20;

\---

&#x20;

\## AI tools disclosure

&#x20;

We built this project with help from AI development tools. They were used for code generation and debugging, application structure, data analysis logic, UI work, documentation, and testing support.

&#x20;

The team reviewed and integrated the final application, architecture, testing and project decisions.

&#x20;

\---

&#x20;

\## Team

&#x20;

\*\*DataForge AI\*\*

&#x20;

\- Ridhi Raheja, AI and Data Science Lead

\- Shiven Pratap Singh, Full-Stack and Backend Developer

\*\*Project:\*\* InsightPilot AI

\*\*Theme:\*\* AI Decision Engine for Business Data

\*\*Problem Statement:\*\* PS-04

&#x20;

\*Turning Business Data into Explainable Decisions\*

&#x20;



