import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Page Configuration ---
st.set_page_config(
    page_title="India Land Use Dashboard",
    page_icon="🌾",
    layout="wide",  # Use wide layout for better dashboard feel
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
# Cache the data loading and preprocessing to speed up the app
@st.cache_data
def load_data(file_path='data.csv'):
    df = pd.read_csv(file_path)

    # Define numeric columns based on your analysis
    numeric_cols = [
        'Reported land area ',
        'Forest land area',
        'Net sown land area',
        'Cropped land area',
        'Land area sown more than once',
        'Barren and unculturable land area',
        'Culturable waste land area'
    ]
    # Include other columns needed for context or filtering
    other_cols = ['State', 'District', 'YearCode', 'Year']

    # Ensure numeric consistency and handle potential errors
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace 0 with NaN in numeric area columns
    df[numeric_cols] = df[numeric_cols].replace(0, np.nan)

    # --- Imputation Strategy ---
    # Group by State and YearCode for more targeted imputation (Optional, Mean is simpler)
    # For simplicity here, using overall column mean as in your original code
    missing_before_imputation = df[numeric_cols].isnull().sum()
    for col in numeric_cols:
         if col in df.columns:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
    missing_after_imputation = df[numeric_cols].isnull().sum()


    # Feature Engineering: Calculate Cropping Intensity (Handle potential division by zero/NaN)
    if 'Cropped land area' in df.columns and 'Net sown land area' in df.columns:
         # Ensure Net sown land area isn't NaN or zero before division
        df['Net sown land area_safe'] = df['Net sown land area'].replace(0, np.nan)
        df['Cropping Intensity'] = df['Cropped land area'] / df['Net sown land area_safe']
        # Fill resulting NaNs in Cropping Intensity (e.g., with 1 or median/mean)
        df['Cropping Intensity'].fillna(df['Cropping Intensity'].median(), inplace=True) # Example: fill with median
        df.drop(columns=['Net sown land area_safe'], inplace=True) # Clean up temp column
    else:
        df['Cropping Intensity'] = np.nan # Assign NaN if columns don't exist

    return df, numeric_cols, missing_before_imputation, missing_after_imputation

# Load the data
df_clean, numeric_cols, missing_before, missing_after = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", [
    "Introduction",
    "National Overview",
    "State-Level Deep Dive",
    "Relationships & Intensity",
    "Data & Methodology"
])

# --- Page Content ---

# === INTRODUCTION PAGE ===
if page == "Introduction":
    st.title("🌾 India Land Use Insights Dashboard")
    st.markdown("---")
    st.subheader("Analyzing Patterns Across States & Time")
    st.markdown("""
        Welcome to the India Land Use Dashboard. This interactive tool presents an analysis of
        district-level land utilization data, primarily focusing on the years **2011-2016**.

        **Why this Analysis?**
        *   Understanding how land is utilized across India is crucial for **agricultural planning**, **resource management**, **environmental policy**, and **sustainable development**.
        *   Variations in land use reflect diverse geographical conditions, agricultural practices, and socio-economic factors.
        *   Analyzing trends and relationships between categories like Net Sown Area, Cropped Area, and Forest Area provides valuable insights for policymakers and researchers.

        **What you'll find:**
        *   A national overview showing overall distributions and statistics.
        *   Detailed comparisons between states.
        *   Analysis of cropping intensity and correlations between land use types.
        *   Information on the data source and methodology used.

        Use the sidebar to navigate through the different sections of the dashboard.
    """)
    st.markdown("---")
    st.header("Key Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("States Covered", df_clean['State'].nunique())
    with col2:
        st.metric("Districts Covered", df_clean['District'].nunique())
    with col3:
        st.metric("Avg. Net Sown Area (Ha)", f"{df_clean['Net sown land area'].mean():,.0f}")
    with col4:
        st.metric("Avg. Cropping Intensity", f"{df_clean['Cropping Intensity'].mean():.2f}")


# === NATIONAL OVERVIEW PAGE ===
elif page == "National Overview":
    st.title("🌍 National Land Use Landscape")
    st.markdown("A high-level view of land use patterns across the districts included in the dataset.")
    st.markdown("---")

    # --- Distribution Plots ---
    st.header("Distribution of Key Land Use Areas")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist_nsa = px.histogram(df_clean, x='Net sown land area', nbins=50,
                                    title='Distribution of Net Sown Area (Hectares)',
                                    labels={'Net sown land area': 'Net Sown Area (Ha)'})
        fig_hist_nsa.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist_nsa, use_container_width=True)
        st.caption("Shows the frequency of districts across different ranges of Net Sown Area.")

    with col2:
        fig_hist_cla = px.histogram(df_clean, x='Cropped land area', nbins=50,
                                     title='Distribution of Cropped Land Area (Hectares)',
                                     labels={'Cropped land area': 'Cropped Land Area (Ha)'})
        fig_hist_cla.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist_cla, use_container_width=True)
        st.caption("Shows the frequency of districts across different ranges of Cropped Land Area.")

    st.markdown("""
        **Observations:**
        *   The distributions for both Net Sown Area and Cropped Land Area appear **right-skewed**. This suggests that while most districts have smaller to moderate areas dedicated to these uses, a few districts have significantly larger areas.
        *   Statistical tests (like Shapiro-Wilk, see Methodology) often confirm these types of distributions are **not perfectly normal**.
    """)
    st.markdown("---")

    # --- National Summary ---
    st.header("National Summary Statistics (Hectares)")
    # Select key columns for the summary table
    summary_cols = ['Reported land area ', 'Forest land area', 'Net sown land area', 'Cropped land area', 'Land area sown more than once', 'Cropping Intensity']
    # Ensure Cropping Intensity is included if calculated
    if 'Cropping Intensity' not in df_clean.columns:
         summary_cols.remove('Cropping Intensity')

    st.dataframe(df_clean[summary_cols].describe().style.format("{:,.2f}"))
    st.caption("Overall mean, standard deviation, min/max, and quartiles for major land categories across all district-year entries.")


# === STATE-LEVEL DEEP DIVE PAGE ===
elif page == "State-Level Deep Dive":
    st.title("🇮🇳 State-Level Land Use Comparison")
    st.markdown("Explore variations and patterns in land use across different Indian states.")
    st.markdown("---")

    # --- State Selection ---
    all_states = sorted(df_clean['State'].unique())
    selected_states = st.multiselect(
        "Select States to Compare (or leave blank for Top 10)",
        all_states,
        default=[] # Start with none selected to show top 10 initially
    )

    if not selected_states:
        # If no states are selected, show top 10 based on a metric (e.g., Net Sown Area)
        top_states_data = df_clean.groupby('State')['Net sown land area'].mean().nlargest(10).reset_index()
        st.subheader("Top 10 States by Average Net Sown Area per District")
        fig_top_states = px.bar(top_states_data, x='State', y='Net sown land area',
                                title="Avg. Net Sown Area (Ha)",
                                labels={'Net sown land area': 'Avg. Net Sown Area (Ha)'})
        fig_top_states.update_layout(xaxis_title=None)
        st.plotly_chart(fig_top_states, use_container_width=True)
        data_to_plot = df_clean[df_clean['State'].isin(top_states_data['State'])]
    else:
        st.subheader(f"Comparison for: {', '.join(selected_states)}")
        data_to_plot = df_clean[df_clean['State'].isin(selected_states)]

    st.markdown("---")

    # --- Box Plots for Variability ---
    st.header("Variability within Selected States")
    col1, col2 = st.columns(2)
    plot_var_1 = 'Net sown land area'
    plot_var_2 = 'Cropping Intensity' # Changed from Cropped land area for variety

    with col1:
        if plot_var_1 in data_to_plot.columns:
            fig_box1 = px.box(data_to_plot, x='State', y=plot_var_1, points="outliers",
                              title=f'{plot_var_1} Distribution by State',
                              labels={plot_var_1: f'{plot_var_1} (Ha)'})
            fig_box1.update_layout(xaxis_title=None)
            st.plotly_chart(fig_box1, use_container_width=True)
            st.caption("Box plots show median (line), interquartile range (box), typical range (whiskers), and potential outliers (dots).")
        else:
            st.warning(f"Column '{plot_var_1}' not found.")

    with col2:
         if plot_var_2 in data_to_plot.columns:
            fig_box2 = px.box(data_to_plot, x='State', y=plot_var_2, points="outliers",
                              title=f'{plot_var_2} Distribution by State',
                              labels={plot_var_2: f'{plot_var_2}'})
            fig_box2.update_layout(xaxis_title=None)
            st.plotly_chart(fig_box2, use_container_width=True)
            st.caption("Cropping intensity reflects how many times land is cropped annually on average.")
         else:
            st.warning(f"Column '{plot_var_2}' not found or not calculated.")

    st.markdown("---")

    # --- Hypothesis Test Result ---
    st.header("Statistical Comparison Example")
    st.markdown("Is there a significant difference in Net Sown Area between Andhra Pradesh and Assam?")

    if 'Andhra Pradesh' in all_states and 'Assam' in all_states:
        andhra = df_clean[df_clean['State'] == 'Andhra Pradesh']['Net sown land area'].dropna()
        assam = df_clean[df_clean['State'] == 'Assam']['Net sown land area'].dropna()

        if len(andhra) > 1 and len(assam) > 1:
            t_stat, p_value = stats.ttest_ind(andhra, assam, equal_var=False, nan_policy='omit') # Handle NaNs just in case
            st.write(f"**T-test Results:**")
            st.write(f"*   T-statistic: {t_stat:.3f}")
            st.write(f"*   P-value: {p_value:.3e}") # Scientific notation for small p-values
            if p_value < 0.05:
                st.success("Conclusion: Reject the null hypothesis. There is a statistically significant difference in the average Net Sown Area between Andhra Pradesh and Assam (p < 0.05).")
            else:
                st.info("Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference found (p >= 0.05).")
        else:
            st.warning("Not enough data for both Andhra Pradesh and Assam to perform T-test.")
    else:
        st.warning("Andhra Pradesh or Assam not found in the dataset.")

# === RELATIONSHIPS & INTENSITY PAGE ===
elif page == "Relationships & Intensity":
    st.title("📈 Land Use Intensity & Correlations")
    st.markdown("Analyzing how intensively land is used and how different land categories relate to each other.")
    st.markdown("---")

    # --- Cropping Intensity ---
    st.header("Cropping Intensity Analysis")
    if 'Cropping Intensity' in df_clean.columns:
        avg_intensity = df_clean.groupby('State')['Cropping Intensity'].mean().reset_index().sort_values(by='Cropping Intensity', ascending=False)
        fig_intensity = px.bar(avg_intensity.head(15), x='State', y='Cropping Intensity',
                               title='Top 15 States by Average Cropping Intensity',
                               labels={'Cropping Intensity': 'Avg. Cropping Intensity'})
        st.plotly_chart(fig_intensity, use_container_width=True)
        st.markdown("""
            **Cropping Intensity** (Calculated as Total Cropped Area / Net Sown Area) indicates how many times, on average, the sown land is cultivated within an agricultural year.
            *   Values **greater than 1** suggest multiple cropping practices (more than one harvest per year on the same land).
            *   This metric is influenced by factors like irrigation availability, climate suitability, soil fertility, and agricultural technology adoption. States like Punjab and Haryana often show high intensity due to extensive irrigation.
        """)
    else:
         st.warning("Cropping Intensity column not available for analysis.")

    st.markdown("---")

    # --- Correlation Analysis ---
    st.header("Correlation Between Land Use Variables")
    st.markdown("Correlation measures the strength and direction of a linear relationship between two variables (-1 to +1).")

    # Select columns for correlation matrix
    corr_cols = numeric_cols + ['Cropping Intensity']
    # Remove Cropping Intensity if it wasn't calculated
    if 'Cropping Intensity' not in df_clean.columns:
        corr_cols.remove('Cropping Intensity')

    correlation_matrix = df_clean[corr_cols].corr()

    # Plot heatmap using Plotly for interactivity
    fig_heatmap = px.imshow(correlation_matrix,
                            text_auto=True, # Show values on heatmap
                            aspect="auto",
                            title="Correlation Matrix Heatmap",
                            color_continuous_scale='RdBu_r', # Red-Blue reversed scale
                            zmin=-1, zmax=1) # Set scale limits
    fig_heatmap.update_layout(height=600) # Adjust height if needed
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("""
        **Observations:**
        *   **Strong Positive Correlations:** As expected, `Net sown land area` and `Cropped land area` are very highly correlated. `Land area sown more than once` also shows a strong positive correlation with `Cropped land area`.
        *   **Moderate Correlations:** Observe other moderate relationships, e.g., `Reported land area` with `Net sown land area` and `Barren/Culturable Waste` areas.
        *   **Cropping Intensity:** Shows a positive correlation with `Land area sown more than once`, which is logical.
    """)
    st.markdown("---")

     # --- Scatter Plot Example ---
    st.header("Example Relationship: Net Sown vs. Cropped Area")
    if 'Net sown land area' in df_clean.columns and 'Cropped land area' in df_clean.columns:
        fig_scatter = px.scatter(df_clean.sample(1000), # Sample for performance
                                 x='Net sown land area', y='Cropped land area',
                                 hover_data=['District', 'State'],
                                 title='Net Sown Area vs. Cropped Land Area (Sample of 1000 points)',
                                 labels={'Net sown land area': 'Net Sown Area (Ha)', 'Cropped land area': 'Cropped Land Area (Ha)'},
                                 opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Each point represents a district-year entry. This visualizes the strong positive correlation.")
    else:
         st.warning("Required columns for scatter plot not found.")


# === DATA & METHODOLOGY PAGE ===
elif page == "Data & Methodology":
    st.title("📊 Data Source and Methodology")
    st.markdown("---")

    st.header("Data Source")
    st.markdown("""
        *   The analysis uses district-level land use data for India, primarily covering the agricultural years **2011-2016** (as indicated by the `YearCode` and `Year` columns).
        *   The dataset includes various categories of land utilization reported in hectares.
        *   Original Source (if known, add link or name here): [e.g., Ministry of Agriculture & Farmers Welfare, Government of India / Specific Portal]
    """)
    st.markdown("---")

    st.header("Data Cleaning and Preparation")
    st.markdown(f"""
        1.  **Column Selection:** Focused on the following numeric columns for area analysis:
            *   `{', '.join(numeric_cols)}`
        2.  **Data Type Conversion:** Ensured selected columns were numeric, coercing errors to NaN.
        3.  **Handling Zeros:** Zero values in the numeric area columns were treated as missing data (replaced with NaN) as zero area is often indicative of missing data rather than true zero.
        4.  **Imputation:** Missing values (NaNs) in the selected numeric columns were filled using the **mean** of each respective column.
            *   *Missing values before imputation:*
                ```
                {missing_before[missing_before > 0].to_string()}
                ```
            *   *Missing values after imputation:* `{missing_after.sum()}` (Should be 0 for imputed columns).
        5.  **Feature Engineering:** Calculated 'Cropping Intensity' as `Cropped land area / Net sown land area`. Handled potential division by zero or NaN results by imputing with the median intensity.
    """)
    st.markdown("---")

    st.header("Analysis Techniques")
    st.markdown("""
        *   **Descriptive Statistics:** Calculated mean, median, standard deviation, min, max, and quartiles.
        *   **Data Visualization:** Used histograms, boxplots, heatmaps, and scatter plots (using Plotly and Seaborn).
        *   **Hypothesis Testing:** Employed an independent samples T-test (Welch's t-test due to potentially unequal variances) to compare means between states.
        *   **Correlation Analysis:** Calculated Pearson correlation coefficients to assess linear relationships.
        *   **Outlier Detection:** Used the Interquartile Range (IQR) method (Outlier if < Q1 - 1.5*IQR or > Q3 + 1.5*IQR).
        *   **Normality Testing:** Used the Shapiro-Wilk test to assess if data follows a normal distribution.
        *   **Multicollinearity Check:** Calculated Variance Inflation Factors (VIF).
    """)
    st.markdown("---")

    st.header("Multicollinearity Assessment (VIF)")
    # Calculate VIF again here or load from cache if needed
    X = df_clean[numeric_cols].dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    st.dataframe(vif_data.style.format({'VIF': "{:.2f}"}))
    st.markdown("""
        **Variance Inflation Factor (VIF)** measures how much the variance of an estimated regression coefficient increases if your predictors are correlated.
        *   A VIF > 5 or 10 often indicates problematic multicollinearity.
        *   **Observation:** `Net sown land area` and `Cropped land area` show extremely high VIF values. This confirms they are highly correlated and capture very similar information, which could cause issues in regression models if both are included as independent variables. `Reported land area` also shows a high VIF, likely due to its correlation with sown/cropped areas.
    """)
    st.markdown("---")

    st.header("Outlier Analysis Example (Net Sown Area)")
    # Calculate outliers again or use stored results
    Q1 = df_clean['Net sown land area'].quantile(0.25)
    Q3 = df_clean['Net sown land area'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_df = df_clean[(df_clean['Net sown land area'] < lower_bound) | (df_clean['Net sown land area'] > upper_bound)]

    st.write(f"Detected **{len(outliers_df)}** potential outliers in 'Net sown land area' using the IQR method (Lower bound: {lower_bound:,.0f}, Upper bound: {upper_bound:,.0f}).")
    st.write("Sample of potential outliers:")
    st.dataframe(outliers_df[['State', 'District', 'YearCode', 'Net sown land area']].head().style.format({'Net sown land area': "{:,.0f}"}))
    st.caption("These represent districts with unusually high or low Net Sown Area compared to the overall distribution.")

    st.markdown("---")
    st.header("Limitations")
    st.markdown("""
        *   The analysis is based on data available for specific years (approx. 2011-2016). Trends may have changed since.
        *   Data accuracy depends on the original source and reporting methods.
        *   Mean imputation was used for missing values, which might slightly distort distributions and reduce variance.
        *   Outlier identification is statistical; domain knowledge is needed to confirm if they are errors or genuine extremes.
    """)


# --- Footer or Common Elements ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard created using Streamlit based on data analysis of Indian Land Use.")