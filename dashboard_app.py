import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
import os
import google.generativeai as genai

st.set_page_config(page_title="📊 CodHelp Assistant Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("CodHelp Dashboard")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload Data File",
    type=["csv", "xls", "xlsx", "json", "txt"]
)


# Load API key from .env or apikey.json
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    import json
    try:
        with open("apikey.json", "r") as f:
            config = json.load(f)
            api_key = config[0].get("api_key")
    except Exception:
        api_key = None

if api_key:
    openai.api_key = api_key

# Load Gemini API key from .env or apikey.json
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    import json
    try:
        with open("apikey.json", "r") as f:
            config = json.load(f)
            gemini_api_key = config[0].get("gemini_api_key")
    except Exception:
        gemini_api_key = None

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

st.title("📊 CodHelp Assistant Dashboard")
st.markdown("<div style='text-align: left; font-size: 12px; color: #888;'>Developed by Vikas Singh</div>", unsafe_allow_html=True)

if uploaded_file:
    # Detect file type and load accordingly
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    elif file_name.endswith(".json"):
        try:
            # Try reading as records (list of dicts)
            df = pd.read_json(uploaded_file, orient="records")
        except Exception:
            uploaded_file.seek(0)
            try:
                # Try reading as lines (newline-delimited JSON)
                df = pd.read_json(uploaded_file, lines=True)
            except Exception:
                uploaded_file.seek(0)
                st.warning("Could not parse JSON as table. Showing raw text below.")
                st.text(uploaded_file.read().decode("utf-8"))
                st.stop()
    elif file_name.endswith(".txt"):
        # Try to read as CSV, fallback to displaying raw text
        try:
            df = pd.read_csv(uploaded_file, delimiter=None)
        except Exception:
            uploaded_file.seek(0)
            st.warning("Could not parse TXT as table. Showing raw text below.")
            st.text(uploaded_file.read().decode("utf-8"))
            st.stop()
    else:
        st.error("Unsupported file format.")
        st.stop()

    orig_df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Filters
    with st.sidebar.expander("Filters", expanded=True):
        filter_cols = st.multiselect("Filter by column", options=cat_cols)
        filtered_df = df.copy()
        for col in filter_cols:
            vals = st.multiselect(f"{col} values", options=filtered_df[col].unique())
            if vals:
                filtered_df = filtered_df[filtered_df[col].isin(vals)]
    if filter_cols:
        df = filtered_df

    # Data Cleaning/Refining Section
    st.header("🧹 Data Cleaning & Refinement")
    before_rows = df.shape[0]
    before_missing = df.isnull().sum().sum()
    before_duplicates = df.duplicated().sum()
    if st.button("Clean & Refine Data"):
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        st.success("Missing values and duplicates removed. Data refined!")
    after_rows = df.shape[0]
    after_missing = df.isnull().sum().sum()
    after_duplicates = df.duplicated().sum()
    st.write(f"**Before:** Rows: {before_rows}, Missing: {before_missing}, Duplicates: {before_duplicates}")
    st.write(f"**After:** Rows: {after_rows}, Missing: {after_missing}, Duplicates: {after_duplicates}")

    # Tabs for dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visuals", "Custom Chart", "AI Insights"])

    # Overview Tab
    with tab1:
        st.header("📈 Data Overview")
        st.subheader("Column Headers")
        st.write(list(df.columns))
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Rows", f"{df.shape[0]:,}")
        kpi2.metric("Columns", f"{df.shape[1]:,}")
        kpi3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        kpi4.metric("Duplicates", f"{df.duplicated().sum():,}")

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🧾 Column Info")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Nulls": df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)

        st.subheader("📊 Basic Statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        # --- Added: Statistics Section ---
        st.subheader("📐 Statistics (Mean, Median, Mode)")
        stat_col = st.selectbox("Select column for statistics", df.columns)
        if pd.api.types.is_numeric_dtype(df[stat_col]):
            mean_val = df[stat_col].mean()
            median_val = df[stat_col].median()
            mode_val = df[stat_col].mode().iloc[0] if not df[stat_col].mode().empty else None
            st.write(f"**Mean:** {mean_val}")
            st.write(f"**Median:** {median_val}")
            st.write(f"**Mode:** {mode_val}")
        else:
            mode_val = df[stat_col].mode().iloc[0] if not df[stat_col].mode().empty else None
            st.write(f"**Mode:** {mode_val}")
            st.info("Mean and median are only available for numeric columns.")

    # Visuals Tab
    with tab2:
        st.header("🎨 Visualizations")
        grid1, grid2 = st.columns(2)
        # Numeric distributions
        with grid1:
            st.subheader("Numeric Column Distribution")
            for col in num_cols:
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", color_discrete_sequence=["#2E86C1"])
                st.plotly_chart(fig, use_container_width=True)
                # Explanation for numeric column
                mean_val = df[col].mean()
                median_val = df[col].median()
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
                st.markdown(
                    f"**{col}**: Mean = `{mean_val:.2f}`, Median = `{median_val:.2f}`, Mode = `{mode_val}`. "
                    "This histogram shows the distribution of values. Peaks indicate common values, while spread shows variability."
                )

        # Top categories
        with grid2:
            st.subheader("Top Categories")
            for col in cat_cols:
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(value_counts, title=f"Top Categories in {col}", labels={"index": col, "value": "Count"}, color_discrete_sequence=["#F5B041"])
                st.plotly_chart(fig, use_container_width=True)
                # Explanation for categorical column
                top_cat = value_counts.index[0] if not value_counts.empty else None
                top_count = value_counts.iloc[0] if not value_counts.empty else None
                st.markdown(
                    f"**{col}**: Most frequent value is `{top_cat}` with `{top_count}` occurrences. "
                    "This bar chart shows the frequency of the top categories, helping identify dominant groups."
                )

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            # Explanation for correlation
            high_corr = corr.abs().unstack().sort_values(ascending=False)
            high_corr = high_corr[high_corr < 1.0]
            if not high_corr.empty:
                top_pair = high_corr.index[0]
                top_val = high_corr.iloc[0]
                st.markdown(
                    f"Highest correlation is between `{top_pair[0]}` and `{top_pair[1]}`: `{top_val:.2f}`. "
                    "Strong correlations suggest a relationship between variables, useful for feature selection and analysis."
                )
            else:
                st.markdown("No significant correlations found between numeric columns.")

        # Pairplot for deeper analysis
        if len(num_cols) > 1:
            st.subheader("Pairplot")
            try:
                import plotly.figure_factory as ff
                fig = ff.create_scatterplotmatrix(df[num_cols], diag='box', index=df.index, colormap='Portland', height=800, width=800)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "Pairplot visualizes pairwise relationships between numeric columns. "
                    "Diagonal boxes show distributions, while off-diagonal plots show scatter relationships."
                )
            except Exception:
                st.info("Pairplot not available for this dataset.")

    # Custom Chart Tab
    with tab3:
        st.header("📊 Custom Interactive Chart")
        chart_type = st.selectbox("Select chart type", ["Bar", "Pie", "Line", "Scatter", "Box", "Histogram"])
        x_col = st.selectbox("X-axis column", df.columns)
        y_col = st.selectbox("Y-axis column (for applicable charts)", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] + ["None"])
        color_col = st.selectbox("Color/Category column (optional)", ["None"] + list(df.columns))

        fig = None
        chart_code = ""
        if chart_type == "Bar":
            if y_col != "None":
                chart_code = f"px.bar(df, x='{x_col}', y='{y_col}', color='{color_col if color_col != 'None' else None}', title='Bar Chart: {x_col} vs {y_col}')"
                fig = px.bar(df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=f"Bar Chart: {x_col} vs {y_col}")
            else:
                chart_code = f"px.bar(df['{x_col}'].value_counts(), title='Bar Chart: {x_col} Counts')"
                fig = px.bar(df[x_col].value_counts(), title=f"Bar Chart: {x_col} Counts")
        elif chart_type == "Pie":
            chart_code = f"px.pie(df, names='{x_col}', title='Pie Chart: {x_col}')"
            fig = px.pie(df, names=x_col, title=f"Pie Chart: {x_col}")
        elif chart_type == "Line":
            if y_col != "None":
                chart_code = f"px.line(df, x='{x_col}', y='{y_col}', color='{color_col if color_col != 'None' else None}', title='Line Chart: {x_col} vs {y_col}')"
                fig = px.line(df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=f"Line Chart: {x_col} vs {y_col}")
        elif chart_type == "Scatter":
            if y_col != "None":
                chart_code = f"px.scatter(df, x='{x_col}', y='{y_col}', color='{color_col if color_col != 'None' else None}', title='Scatter Plot: {x_col} vs {y_col}')"
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=f"Scatter Plot: {x_col} vs {y_col}")
        elif chart_type == "Box":
            if y_col != "None":
                chart_code = f"px.box(df, x='{x_col}', y='{y_col}', color='{color_col if color_col != 'None' else None}', title='Box Plot: {x_col} vs {y_col}')"
                fig = px.box(df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=f"Box Plot: {x_col} vs {y_col}")
        elif chart_type == "Histogram":
            chart_code = f"px.histogram(df, x='{x_col}', color='{color_col if color_col != 'None' else None}', title='Histogram: {x_col}')"
            fig = px.histogram(df, x=x_col, color=color_col if color_col != "None" else None, title=f"Histogram: {x_col}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Chart Code")
            st.code(chart_code, language="python")
            # Editable code area
            st.markdown("#### Edit and Run Chart Code")
            user_code = st.text_area("Edit the code below and click 'Run Chart'", value=chart_code, height=100)
            if st.button("Run Chart"):
                try:
                    local_vars = {"df": df, "px": px, "plt": plt, "sns": sns, "pd": pd}
                    custom_fig = eval(user_code, {}, local_vars)
                    st.plotly_chart(custom_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in custom chart code: {e}")
        else:
            st.info("Select appropriate columns for the chosen chart type.")

        # --- New Feature: Project Code Editor ---
        st.markdown("---")
        st.header("📝 Project Code Editor")
        st.markdown("Type any Python code to analyze or visualize your data. You can use `df`, `px`, `plt`, `sns`, `pd` in your code. Output will be shown below.")
        code_input = st.text_area(
            "Write your dashboard code here (like Colab):",
            value="import plotly.express as px\nfig = px.scatter(df, x=df.columns[0], y=df.columns[1])\nst.plotly_chart(fig)",
            height=200
        )
        if st.button("Run Project Code"):
            import io
            import contextlib
            output_buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(code_input, {"df": df, "px": px, "plt": plt, "sns": sns, "pd": pd, "st": st})
                st.success("Code executed successfully.")
                output = output_buffer.getvalue()
                if output:
                    st.text(output)
            except Exception as e:
                st.error(f"Error in project code: {e}")

#     # AI Insights Tab
#     with tab4:
#         st.header("🤖 AI Insights & Suggestions (Gemini)")
#         if gemini_api_key:
#             prompt = f"Analyze the following data and provide key insights, trends, and suggestions for visualization:\n\n{df.head(20).to_csv(index=False)}"
#             try:
#                 model = genai.GenerativeModel("gemini-pro")
#                 response = model.generate_content(prompt)
#                 st.info(response.text)
#             except Exception as e:
#                 st.error(f"Gemini API error: {e}")

#             # Custom Gemini Q&A
#             st.subheader("💬 Ask Gemini AI about your data")
#             user_question = st.text_input("Type your question about the data (e.g., 'What are the main trends?', 'Which column is most important?')")
#             if user_question:
#                 custom_prompt = f"Given this data:\n{df.head(20).to_csv(index=False)}\n\nAnswer this question: {user_question}"
#                 try:
#                     response = model.generate_content(custom_prompt)
#                     st.success(response.text)
#                 except Exception as e:
#                     st.error(f"Gemini API error: {e}")
#         else:
#             st.info("Gemini API key not found. Add it to .env or apikey.json as GEMINI_API_KEY for AI-powered insights.")

# else:
#     st.info("Please upload a CSV file using the sidebar to begin.")
