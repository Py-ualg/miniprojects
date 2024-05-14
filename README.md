# Miniprojects
Quick coding experiments and small projects with some real outputs.

You can open the repository in server-hosted online environment, [Binder](https://mybinder.org/) from here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Py-ualg/miniprojects/HEAD)

## Description of projects:
1. alcoholism
    * Promoting dry January, we developed full pipeline of basic data-science model predicting alcohol consumption from publically available data.
    * First get the data from World bank using python API, `wbgapi`. Second clean and do preliminary analysis using `pandas`, together with visualization in `seaborn`. Third, develop linear regression model in `scikit-learn`, explore spurious correlation, overfitting, etc.

2. streamlit-dashboard
    * Minimalistic way how to create `streamlit` interactive dashboard using examplery ML Iris dataset. 
    * You see that in 130 lines of code, you train model, visualize results, and create interactive environment for the user to look into your KMeans clustering.

3. r2py
    * In 2022, introductory course to R has been given by [@ramiromagno](https://github.com/ramiromagno) and [@iduarte](https://github.com/iduarte) at CCMAR using cute synthetic *crab* dataset. All resources in R can be found [here](https://rmagno.eu/tdvr.oct.22/).
    * This project translates and compares how the same things are done using python, specifically `pandas`.


## Geeksessions 2024 May 22
1. Clone the miniprojects repository
2. Go to [streamlit](https://streamlit.io)
3. Sign in with GH, gmail or other
    * If you want to run dashboard from your organization, follow instructions [here](https://docs.github.com/en/organizations/managing-oauth-access-to-your-organizations-data/about-oauth-app-access-restrictions)
4. Create app
    * repo: .../miniprojects
    * branch: geeks-240522
    * Main file-path: streamlit_dashboard/app.py
    * app URL: UP-TO-YOU
5. You should have the app functional and running

### Running locally
* `pip install streamlit`
* `streamlit run streamlit_dashboard/app.py`

