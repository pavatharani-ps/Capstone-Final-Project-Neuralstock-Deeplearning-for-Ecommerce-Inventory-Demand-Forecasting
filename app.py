import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="NeuralStock Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📦 NeuralStock Dashboard")

st.markdown(
    """
    ### AI-Based Inventory Demand Forecasting System
    """
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    # ---------------------------------------------------
    # DATASET PREVIEW
    # ---------------------------------------------------

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # NUMERIC COLUMNS
    # ---------------------------------------------------

    # Select only integer and float columns
    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Remove ID columns if present
    numeric_columns = [
        col for col in numeric_columns
        if 'id' not in col.lower()
    ]

    if len(numeric_columns) == 0:

        st.error("❌ No numeric columns found")

    else:

        # Select first numeric column as sales
        sales_column = numeric_columns[0]

        st.info(
            f"Using '{sales_column}' as Sales Column"
        )

        # ---------------------------------------------------
        # CATEGORY FILTER
        # ---------------------------------------------------

        category_columns = df.select_dtypes(
            include='object'
        ).columns

        filtered_df = df.copy()

        if len(category_columns) > 0:

            category_column = category_columns[0]

            categories = df[
                category_column
            ].dropna().unique()

            selected_category = st.sidebar.selectbox(
                "📂 Select Category",
                categories
            )

            filtered_df = filtered_df[
                filtered_df[category_column]
                == selected_category
            ]

        # ---------------------------------------------------
        # DATE FILTER
        # ---------------------------------------------------

        date_columns = []

        # Only check object columns for dates
        object_columns = filtered_df.select_dtypes(
            include='object'
        ).columns

        for col in object_columns:

            try:

                converted_col = pd.to_datetime(
                    filtered_df[col],
                    errors='coerce'
                )

                # If most values are valid dates
                if converted_col.notnull().sum() > 0:

                    filtered_df[col] = converted_col

                    date_columns.append(col)

            except:

                pass

        if len(date_columns) > 0:

            date_column = date_columns[0]

            filtered_df = filtered_df.dropna(
                subset=[date_column]
            )

            min_date = filtered_df[
                date_column
            ].min()

            max_date = filtered_df[
                date_column
            ].max()

            date_range = st.sidebar.date_input(
                "📅 Select Date Range",
                [min_date, max_date]
            )

            if len(date_range) == 2:

                start_date = pd.to_datetime(
                    date_range[0]
                )

                end_date = pd.to_datetime(
                    date_range[1]
                )

                filtered_df = filtered_df[
                    (
                        filtered_df[date_column]
                        >= start_date
                    )
                    &
                    (
                        filtered_df[date_column]
                        <= end_date
                    )
                ]
                
        # ---------------------------------------------------
        # KPI CARDS
        # ---------------------------------------------------

        total_sales = filtered_df[
            sales_column
        ].sum()

        avg_sales = filtered_df[
            sales_column
        ].mean()

        max_sales = filtered_df[
            sales_column
        ].max()

        min_sales = filtered_df[
            sales_column
        ].min()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Total Sales",
                f"{total_sales:.2f}"
            )

        with col2:
            st.metric(
                "📊 Average Sales",
                f"{avg_sales:.2f}"
            )

        with col3:
            st.metric(
                "📈 Maximum Sales",
                f"{max_sales:.2f}"
            )

        with col4:
            st.metric(
                "📉 Minimum Sales",
                f"{min_sales:.2f}"
            )

        # ---------------------------------------------------
        # SALES TREND CHART
        # ---------------------------------------------------

        st.subheader("📈 Sales Trend Forecast")

        fig = px.line(
            filtered_df,
            y=sales_column,
            title="Inventory Demand Forecast"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------------------------
        # HISTOGRAM
        # ---------------------------------------------------

        st.subheader("📊 Sales Distribution")

        hist_fig = px.histogram(
            filtered_df,
            x=sales_column,
            nbins=30,
            title="Sales Distribution"
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

        # ---------------------------------------------------
        # REORDER ALERT
        # ---------------------------------------------------

        st.subheader("🚨 Reorder Alert")

        threshold = st.slider(
            "Select Reorder Threshold",
            min_value=1,
            max_value=100,
            value=60
        )

        low_stock = filtered_df[
            filtered_df[sales_column]
            < threshold
        ]

        if len(low_stock) > 0:

            st.warning(
                f"⚠️ {len(low_stock)} low stock records found"
            )

            st.dataframe(low_stock)

        else:

            st.success(
                "✅ No low stock alerts"
            )

        # ---------------------------------------------------
        # FORECAST STATISTICS
        # ---------------------------------------------------

        st.subheader("📋 Forecast Statistics")

        st.dataframe(
            filtered_df[
                sales_column
            ].describe()
        )

        # ---------------------------------------------------
        # DOWNLOAD BUTTON
        # ---------------------------------------------------

        st.download_button(
            label="⬇️ Download Forecast Data",
            data=filtered_df.to_csv(
                index=False
            ),
            file_name="forecast_data.csv",
            mime="text/csv"
        )

else:

    st.info(
        "📂 Please Upload a CSV File"
    )