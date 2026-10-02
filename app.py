import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from google import genai
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightPilot AI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #f5f7fa;
        margin-bottom: 10px;
    }

    .small-text {
        color: #777;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 InsightPilot AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Turning Business Data into Explainable Decisions</div>',
    unsafe_allow_html=True
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ InsightPilot AI")

st.sidebar.markdown(
    """
    **AI Decision Engine**

    InsightPilot AI helps you:

    - Understand business data
    - Calculate KPIs
    - Detect trends
    - Find anomalies
    - Analyze correlations
    - Ask business questions
    - Get evidence-based insights
    - Receive recommendations
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Upload your dataset using the upload box on the main page."
)


# ============================================================
# MAIN PAGE FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📁 Upload Business Data</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX and XLS"
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV or Excel file above to start your analysis."
    )

    st.markdown(
        """
        ### How it works

        **1. Upload Data → 2. Analyze → 3. Ask Questions → 4. Get Decisions**

        InsightPilot AI combines deterministic data analysis with
        generative AI to provide explainable business insights.
        """
    )

    st.markdown(
        """
        ### What you can do

        | Capability | Description |
        |---|---|
        | 📊 Data Understanding | Automatically understand your dataset |
        | 📈 KPI Analysis | Calculate important numerical metrics |
        | 📉 Trend Detection | Identify changes over time |
        | 🚨 Anomaly Detection | Detect unusual records |
        | 🔗 Correlation Analysis | Find relationships between variables |
        | 🤖 AI Questions | Ask questions in natural language |
        | 🔍 Evidence | See the data supporting the answer |
        | 💡 Recommendations | Get business-oriented next steps |
        """
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Could not read the uploaded file: {e}"
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

df.columns = df.columns.astype(str).str.strip()

df = df.dropna(how="all")

df = df.dropna(
    axis=1,
    how="all"
)


# ============================================================
# DATA TYPE DETECTION
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# Detect date columns

date_columns = []

for column in df.columns:

    if column in numeric_columns:
        continue

    try:

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.80:

            date_columns.append(column)

    except Exception:
        pass


categorical_columns = [
    col
    for col in categorical_columns
    if col not in date_columns
]


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📋 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:

    st.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

with col3:

    st.metric(
        "Missing Values",
        f"{int(df.isnull().sum().sum()):,}"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "👀 View Dataset Preview",
    expanded=False
):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DATA TYPES
# ============================================================

with st.expander(
    "🔎 Detected Data Types",
    expanded=False
):

    type_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[column].dtype)
            for column in df.columns
        ]
    })

    st.dataframe(
        type_df,
        use_container_width=True
    )

    if numeric_columns:

        st.write("**Numeric columns:**")

        st.write(
            ", ".join(numeric_columns)
        )

    if categorical_columns:

        st.write("**Categorical columns:**")

        st.write(
            ", ".join(categorical_columns)
        )

    if date_columns:

        st.write("**Date columns:**")

        st.write(
            ", ".join(date_columns)
        )


# ============================================================
# DATA QUALITY
# ============================================================

st.markdown(
    '<div class="section-title">🧹 Data Quality</div>',
    unsafe_allow_html=True
)

missing_df = pd.DataFrame({
    "Column": df.columns,

    "Missing Values": [
        int(df[column].isnull().sum())
        for column in df.columns
    ],

    "Missing %": [
        round(
            df[column].isnull().mean() * 100,
            2
        )
        for column in df.columns
    ]
})

st.dataframe(
    missing_df,
    use_container_width=True
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Key Performance Indicators</div>',
    unsafe_allow_html=True
)

if numeric_columns:

    kpi_columns = numeric_columns[:4]

    kpi_cards = st.columns(
        len(kpi_columns)
    )

    for i, column in enumerate(kpi_columns):

        value = df[column].mean()

        with kpi_cards[i]:

            st.metric(
                f"Avg {column}",
                f"{value:,.2f}"
            )

else:

    st.info(
        "No numeric columns were detected."
    )


# ============================================================
# NUMERIC ANALYSIS
# ============================================================

if numeric_columns:

    st.markdown(
        '<div class="section-title">📊 Numeric Analysis</div>',
        unsafe_allow_html=True
    )

    selected_numeric = st.selectbox(
        "Select a numeric column",
        numeric_columns
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_hist = px.histogram(
            df,
            x=selected_numeric,
            title=f"Distribution of {selected_numeric}",
            marginal="box"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with col2:

        summary_values = df[
            selected_numeric
        ].describe()

        summary_df = pd.DataFrame({
            "Statistic": summary_values.index,
            "Value": summary_values.values
        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )


# ============================================================
# CATEGORY COMPARISON
# ============================================================

if categorical_columns and numeric_columns:

    st.markdown(
        '<div class="section-title">📊 Category Comparison</div>',
        unsafe_allow_html=True
    )

    category_column = st.selectbox(
        "Select category",
        categorical_columns
    )

    value_column = st.selectbox(
        "Select numeric value",
        numeric_columns
    )

    category_limit = st.slider(
        "Number of categories to display",
        min_value=5,
        max_value=20,
        value=10
    )

    category_summary = (
        df.groupby(category_column)[value_column]
        .agg(
            mean="mean",
            sum="sum",
            count="count"
        )
        .sort_values(
            "mean",
            ascending=False
        )
        .head(category_limit)
        .reset_index()
    )

    fig_category = px.bar(
        category_summary,
        x=category_column,
        y="mean",
        title=(
            f"Average {value_column} "
            f"by {category_column}"
        ),
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.dataframe(
        category_summary,
        use_container_width=True
    )


# ============================================================
# TREND ANALYSIS
# ============================================================

if date_columns and numeric_columns:

    st.markdown(
        '<div class="section-title">📅 Trend Analysis</div>',
        unsafe_allow_html=True
    )

    trend_date = st.selectbox(
        "Select date column",
        date_columns
    )

    trend_value = st.selectbox(
        "Select value column",
        numeric_columns,
        key="trend_value"
    )

    trend_df = df.copy()

    trend_df[trend_date] = pd.to_datetime(
        trend_df[trend_date],
        errors="coerce"
    )

    trend_df = trend_df.dropna(
        subset=[
            trend_date,
            trend_value
        ]
    )

    trend_df = (
        trend_df
        .groupby(trend_date)[trend_value]
        .sum()
        .reset_index()
        .sort_values(trend_date)
    )

    if len(trend_df) > 1:

        fig_trend = px.line(
            trend_df,
            x=trend_date,
            y=trend_value,
            markers=True,
            title=(
                f"{trend_value} "
                f"Trend Over Time"
            )
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

        first_value = trend_df[
            trend_value
        ].iloc[0]

        last_value = trend_df[
            trend_value
        ].iloc[-1]

        if first_value != 0:

            percentage_change = (
                (
                    last_value - first_value
                )
                / abs(first_value)
            ) * 100

            st.metric(
                "Overall Change",
                f"{percentage_change:.2f}%"
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

if (
    len(numeric_columns) > 0
    and len(df) >= 20
):

    st.markdown(
        '<div class="section-title">🚨 Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    anomaly_column = st.selectbox(
        "Select column for anomaly detection",
        numeric_columns,
        key="anomaly_column"
    )

    anomaly_data = df[
        [anomaly_column]
    ].dropna().copy()

    if len(anomaly_data) >= 20:

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        predictions = model.fit_predict(
            anomaly_data[
                [anomaly_column]
            ]
        )

        anomaly_data["Anomaly"] = predictions

        anomaly_count = int(
            (predictions == -1).sum()
        )

        normal_count = int(
            (predictions == 1).sum()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal Records",
                f"{normal_count:,}"
            )

        with col2:

            st.metric(
                "Potential Anomalies",
                f"{anomaly_count:,}"
            )

        fig_anomaly = px.scatter(
            anomaly_data.reset_index(),
            x="index",
            y=anomaly_column,
            color="Anomaly",
            title=(
                f"Anomaly Detection — "
                f"{anomaly_column}"
            )
        )

        st.plotly_chart(
            fig_anomaly,
            use_container_width=True
        )

    else:

        st.info(
            "Not enough valid records for anomaly detection."
        )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

if len(numeric_columns) >= 2:

    st.markdown(
        '<div class="section-title">🔗 Correlation Analysis</div>',
        unsafe_allow_html=True
    )

    correlation_matrix = df[
        numeric_columns
    ].corr()

    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Numeric Feature Correlation"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# ============================================================
# VERIFIED BUSINESS EVIDENCE
# ============================================================

def generate_business_evidence(dataframe):

    evidence = {}

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    evidence["dataset_overview"] = {
        "rows": int(
            dataframe.shape[0]
        ),

        "columns": int(
            dataframe.shape[1]
        ),

        "missing_values": int(
            dataframe.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            dataframe.duplicated().sum()
        )
    }

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    numeric_cols = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_statistics = {}

    for column in numeric_cols:

        series = dataframe[
            column
        ].dropna()

        if len(series) == 0:
            continue

        numeric_statistics[column] = {
            "count": int(
                series.count()
            ),

            "mean": round(
                float(series.mean()),
                4
            ),

            "median": round(
                float(series.median()),
                4
            ),

            "minimum": round(
                float(series.min()),
                4
            ),

            "maximum": round(
                float(series.max()),
                4
            ),

            "standard_deviation": round(
                float(series.std()),
                4
            )
        }

    evidence[
        "numeric_statistics"
    ] = numeric_statistics

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    categorical_cols = dataframe.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    category_statistics = {}

    for category_column in categorical_cols[:8]:

        unique_count = dataframe[
            category_column
        ].nunique(
            dropna=True
        )

        if unique_count > 50:
            continue

        category_statistics[
            category_column
        ] = {}

        for numeric_column in numeric_cols[:8]:

            try:

                grouped = (
                    dataframe
                    .groupby(
                        category_column
                    )[numeric_column]
                    .agg(
                        mean="mean",
                        sum="sum",
                        count="count"
                    )
                    .sort_values(
                        "mean",
                        ascending=False
                    )
                    .head(10)
                    .reset_index()
                )

                rows = []

                for _, row in grouped.iterrows():

                    rows.append({
                        "category": str(
                            row[
                                category_column
                            ]
                        ),

                        "average": round(
                            float(
                                row["mean"]
                            ),
                            4
                        ),

                        "total": round(
                            float(
                                row["sum"]
                            ),
                            4
                        ),

                        "count": int(
                            row["count"]
                        )
                    })

                category_statistics[
                    category_column
                ][numeric_column] = rows

            except Exception:
                continue

    evidence[
        "category_statistics"
    ] = category_statistics

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_statistics = {}

    for column in dataframe.columns:

        missing_count = int(
            dataframe[
                column
            ].isnull().sum()
        )

        if missing_count > 0:

            missing_statistics[
                column
            ] = {
                "missing_count": missing_count,

                "missing_percentage": round(
                    float(
                        dataframe[
                            column
                        ].isnull().mean()
                        * 100
                    ),
                    2
                )
            }

    evidence[
        "missing_values"
    ] = missing_statistics

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_pairs = []

    if len(numeric_cols) >= 2:

        correlation_matrix = dataframe[
            numeric_cols
        ].corr()

        for i in range(
            len(numeric_cols)
        ):

            for j in range(
                i + 1,
                len(numeric_cols)
            ):

                column_a = numeric_cols[i]
                column_b = numeric_cols[j]

                correlation = (
                    correlation_matrix.loc[
                        column_a,
                        column_b
                    ]
                )

                if pd.notna(
                    correlation
                ):

                    correlation_pairs.append({
                        "column_1": column_a,

                        "column_2": column_b,

                        "correlation": round(
                            float(
                                correlation
                            ),
                            4
                        )
                    })

        correlation_pairs = sorted(
            correlation_pairs,
            key=lambda x: abs(
                x["correlation"]
            ),
            reverse=True
        )[:15]

    evidence[
        "strongest_correlations"
    ] = correlation_pairs

    # --------------------------------------------------------
    # Trend statistics
    # --------------------------------------------------------

    trend_statistics = {}

    detected_dates = []

    for column in dataframe.columns:

        if column in numeric_cols:
            continue

        try:

            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce"
            )

            if converted.notna().mean() >= 0.80:

                detected_dates.append(
                    column
                )

        except Exception:
            pass

    for date_column in detected_dates[:3]:

        converted = pd.to_datetime(
            dataframe[
                date_column
            ],
            errors="coerce"
        )

        temp = dataframe.copy()

        temp["_detected_date"] = converted

        temp = temp.dropna(
            subset=[
                "_detected_date"
            ]
        )

        if len(temp) < 2:
            continue

        trend_statistics[
            date_column
        ] = {}

        for numeric_column in numeric_cols[:8]:

            try:

                grouped = (
                    temp
                    .groupby(
                        "_detected_date"
                    )[numeric_column]
                    .sum()
                    .sort_index()
                )

                if len(grouped) >= 2:

                    first_value = float(
                        grouped.iloc[0]
                    )

                    last_value = float(
                        grouped.iloc[-1]
                    )

                    if first_value != 0:

                        change = (
                            (
                                last_value
                                - first_value
                            )
                            / abs(first_value)
                        ) * 100

                    else:

                        change = None

                    trend_statistics[
                        date_column
                    ][numeric_column] = {

                        "start_date": str(
                            grouped.index[
                                0
                            ].date()
                        ),

                        "end_date": str(
                            grouped.index[
                                -1
                            ].date()
                        ),

                        "starting_value": round(
                            first_value,
                            4
                        ),

                        "ending_value": round(
                            last_value,
                            4
                        ),

                        "percentage_change": (
                            round(
                                change,
                                2
                            )
                            if change is not None
                            else None
                        )
                    }

            except Exception:
                continue

    evidence[
        "trend_statistics"
    ] = trend_statistics

    # --------------------------------------------------------
    # Anomaly statistics
    # --------------------------------------------------------

    anomaly_statistics = {}

    for numeric_column in numeric_cols[:8]:

        series = dataframe[
            numeric_column
        ].dropna()

        if len(series) < 20:
            continue

        try:

            model = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            predictions = model.fit_predict(
                series.to_frame()
            )

            anomaly_count = int(
                (predictions == -1).sum()
            )

            anomaly_statistics[
                numeric_column
            ] = {

                "anomaly_count":
                    anomaly_count,

                "anomaly_percentage":
                    round(
                        anomaly_count
                        / len(series)
                        * 100,
                        2
                    )
            }

        except Exception:
            continue

    evidence[
        "anomalies"
    ] = anomaly_statistics

    return evidence


# ============================================================
# PREPARE EVIDENCE
# ============================================================

with st.spinner(
    "🔍 Preparing verified business evidence..."
):

    business_evidence = (
        generate_business_evidence(df)
    )


# ============================================================
# AI DECISION ENGINE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Decision Engine</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask a business question about your uploaded dataset."
)

question = st.text_area(
    "💬 Your business question",

    placeholder=(
        "Example: Which category has the highest "
        "average value and what action should "
        "the business take?"
    ),

    height=100
)


# ============================================================
# AI ANALYSIS FUNCTION
# ============================================================

def run_ai_analysis(
    user_question,
    evidence
):

    if client is None:

        return (
            None,
            "Gemini API is not configured. "
            "Please check your GEMINI_API_KEY "
            "in .streamlit/secrets.toml."
        )

    evidence_text = str(
        evidence
    )

    prompt = f"""
You are InsightPilot AI, an explainable
business decision engine.

Your job is to analyze verified business
evidence calculated directly from a user's dataset.

IMPORTANT RULES:

1. Use ONLY the verified evidence provided below.
2. Do NOT invent numbers, categories, or findings.
3. Do NOT claim calculations that are not present.
4. If the evidence is insufficient, explicitly say so.
5. Keep numerical values exactly consistent with the evidence.
6. Separate factual findings from business interpretation.
7. Missing values and duplicate rows are DATA QUALITY issues.
8. Anomalies are ANALYTICAL findings, not automatically data-quality errors.
9. Correlations describe relationships; they do not prove causation.
10. Recommendations must be based on the evidence.
11. Do not claim a recommendation is guaranteed to work.
12. Clearly state the actual columns and values supporting your answer.
13. Be concise but useful.

VERIFIED DATASET EVIDENCE:

{evidence_text}

USER'S BUSINESS QUESTION:

{user_question}

Return exactly these sections:

### Answer

Directly answer the user's question.

### Evidence

List the specific dataset facts supporting
the answer.

### Business Interpretation

Explain what the evidence means
from a business perspective.

### Recommendation

Give practical next steps based
on the evidence.

### Confidence

Choose one:

High
Medium
Low

If the evidence is insufficient,
say so instead of guessing.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return (
            response,
            None
        )

    except Exception as e:

        return (
            None,
            str(e)
        )


# ============================================================
# RUN AI ANALYSIS
# ============================================================

if st.button(
    "🚀 Analyze with InsightPilot AI",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a business question first."
        )

    else:

        start_time = time.time()

        with st.spinner(
            "🤖 InsightPilot AI is analyzing "
            "the verified evidence..."
        ):

            response, error = (
                run_ai_analysis(
                    question,
                    business_evidence
                )
            )

        response_time = (
            time.time()
            - start_time
        )

        if error:

            st.error(
                f"❌ AI analysis failed: {error}"
            )

        elif response:

            st.success(
                f"Analysis completed in "
                f"{response_time:.2f} seconds."
            )

            st.markdown(
                "### 💡 AI Decision"
            )

            st.markdown(
                response.text
            )

            st.markdown(
                "### 📌 Decision Summary"
            )

            st.info(
                "The AI recommendation is grounded "
                "in verified statistics calculated "
                "from your dataset."
            )

            st.download_button(
                label="📥 Download AI Analysis",

                data=response.text,

                file_name=(
                    "insightpilot_ai_analysis.txt"
                ),

                mime="text/plain"
            )

            st.caption(
                f"Response time: "
                f"{response_time:.2f} seconds"
            )


# ============================================================
# AUTOMATIC INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Automatic Insights</div>',
    unsafe_allow_html=True
)

automatic_insights = []


# Missing values

total_missing = int(
    df.isnull().sum().sum()
)

if total_missing > 0:

    automatic_insights.append(
        f"The dataset contains "
        f"{total_missing:,} missing values."
    )

else:

    automatic_insights.append(
        "The dataset has no missing values."
    )


# Duplicate rows

duplicates = int(
    df.duplicated().sum()
)

if duplicates > 0:

    automatic_insights.append(
        f"The dataset contains "
        f"{duplicates:,} duplicate rows."
    )

else:

    automatic_insights.append(
        "No duplicate rows were detected."
    )


# Highest average numeric column

if numeric_columns:

    averages = {}

    for column in numeric_columns:

        try:

            averages[column] = (
                df[column].mean()
            )

        except Exception:
            pass

    if averages:

        highest_average_column = max(
            averages,
            key=averages.get
        )

        automatic_insights.append(
            f"Among numeric columns, "
            f"{highest_average_column} has the "
            f"highest average value of "
            f"{averages[highest_average_column]:,.2f}."
        )


# Anomalies

if business_evidence[
    "anomalies"
]:

    highest_anomaly_column = max(
        business_evidence[
            "anomalies"
        ],

        key=lambda column:
        business_evidence[
            "anomalies"
        ][column][
            "anomaly_percentage"
        ]
    )

    anomaly_percentage = (
        business_evidence[
            "anomalies"
        ][
            highest_anomaly_column
        ][
            "anomaly_percentage"
        ]
    )

    automatic_insights.append(
        f"{highest_anomaly_column} has approximately "
        f"{anomaly_percentage:.2f}% potential anomalies "
        f"according to Isolation Forest."
    )


for insight in automatic_insights:

    st.markdown(
        f"""
        <div class="insight-box">
        💡 {insight}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NUMERIC SUMMARY
# ============================================================

if numeric_columns:

    st.markdown(
        '<div class="section-title">📊 Numeric Summary</div>',
        unsafe_allow_html=True
    )

    numeric_summary = df[
        numeric_columns
    ].describe().T

    numeric_summary = (
        numeric_summary.round(2)
    )

    st.dataframe(
        numeric_summary,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightPilot AI — AI-powered decision "
    "support for business data."
)

st.caption(
    "AI-generated recommendations should be "
    "reviewed by a human before making "
    "business decisions."
)