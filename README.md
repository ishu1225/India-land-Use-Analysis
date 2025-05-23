Land Use Data Analysis in India

Exploring land use patterns across Indian districts (2011–2015) through data analysis and visualization. 


📋 Table of Contents





Project Overview



Objectives



Key Findings



Dataset



Requirements



Setup Instructions



Project Structure



Technologies Used



Screenshots



Future Work



Contributing

🌍 Project Overview

This Python-based data analysis project investigates land use patterns across Indian districts from 2011 to 2015. The dataset, comprising 5,786 records, includes metrics such as Net sown land area, Cropped land area, Forest land area, and more. Through data cleaning, statistical analysis, and visualization, the project uncovers insights into agricultural and environmental land use trends.

🎯 Objectives





Clean and preprocess land use data for robust analysis.



Summarize key metrics using descriptive statistics (e.g., median, mode).



Compare agricultural land use across states (e.g., Andhra Pradesh vs. Assam).



Analyze variable distributions and test for normality.



Assess multicollinearity among land use metrics.



Visualize findings for enhanced interpretability.

🔍 Key Findings





Successfully cleaned the dataset, imputing missing values and handling zeros.



Identified significant differences in Net sown land area between Andhra Pradesh and Assam via t-test.



Confirmed non-normal distribution of Net sown land area using Shapiro-Wilk tests and histograms.



Detected high multicollinearity (e.g., VIF 2,982.59 for Cropped land area) among agricultural variables.



Visualized distributions with kernel density estimates and normal fit curves.

📊 Dataset

The dataset (data.csv) contains land use data across Indian districts, with columns for geographical identifiers (e.g., State, District), temporal data (Year), and land use metrics.
Note: The dataset is not included in this repository due to size or licensing constraints. Users can replace it with a similar dataset or contact the author for details.

🛠️ Requirements





Python: 3.8–3.10 (developed with 3.13.2, but older versions recommended for compatibility)



Required Packages:

pip install pandas numpy scipy statsmodels seaborn matplotlib

🚀 Setup Instructions





Clone the repository:

git clone https://github.com/your-username/land-use-analysis-india.git
cd land-use-analysis-india



Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies:

pip install -r requirements.txt



Place data.csv in the project root or update the file path in the notebook.



Run the Jupyter Notebook:

jupyter notebook Land_Use_Analysis.ipynb

📂 Project Structure





Land_Use_Analysis.ipynb: Main Jupyter Notebook with data cleaning, analysis, and visualizations.



data.csv: Placeholder for the dataset (not included; user must provide).



requirements.txt: List of required Python packages.



README.md: This file.

💻 Technologies Used





Python: Core programming language.



pandas & numpy: Data manipulation and numerical operations.



scipy & statsmodels: Statistical testing and VIF analysis.



seaborn & matplotlib: Data visualization.

📸 Screenshots



🔮 Future Work





Implement regression models to predict land use trends.



Analyze temporal changes across years.



Develop an interactive Streamlit dashboard for visualizations.



Incorporate additional datasets (e.g., climate, crop yield).

🤝 Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a feature branch:

git checkout -b feature-branch



Commit changes:

git commit -m "Add feature"



Push to the branch:

git push origin feature-branch



Open a pull request.
