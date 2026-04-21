# IRT Parameter Calibration Web Application

A comprehensive Streamlit web application for Item Response Theory (IRT) parameter calibration using the 2-Parameter Logistic (2PL) model with the EM algorithm.

## Features

### 🎯 Core Functionality
- **Data Upload**: Support for CSV and Excel files containing binary response matrices
- **EM Algorithm**: Classical Expectation-Maximization algorithm for 2PL model calibration
- **Gaussian Quadrature**: Numerical integration using fixed quadrature points (-4 to 4)
- **Parameter Estimation**: 
  - Item discrimination (a) parameters
  - Item difficulty (b) parameters
  - Standard errors for both parameters
- **Person Scoring**: EAP (Expected A Posteriori) ability estimation with standard errors

### 📊 Visualizations
- **Item Characteristic Curves (ICC)**: Individual or combined view of all items
- **Test Information Function (TIF)**: Overall test precision across ability levels
- **Parameter Tables**: Comprehensive display with standard errors

### 🎨 User Interface
- **Sidebar Settings**: 
  - Number of quadrature points (10-80)
  - Convergence tolerance (1e-5 to 1e-2)
  - Maximum iterations
- **Tabbed Interface**:
  - Data Preview: View response matrix and item statistics
  - Item Parameters: Run calibration and view results
  - Visualizations: Interactive plots
- **Download Options**: Export parameters and person scores as CSV

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run irt_calibration_app.py
```

2. Open your browser to the provided URL (typically `http://localhost:8501`)

3. Upload your data file:
   - Format: CSV or Excel
   - Structure: Rows = persons, Columns = items
   - Values: 0 (incorrect) or 1 (correct)

4. Adjust settings in the sidebar if needed

5. Navigate through the tabs:
   - **Data Preview**: Review your data and item statistics
   - **Item Parameters**: Click "Run Calibration" to estimate parameters
   - **Visualizations**: Explore ICCs and TIF

## Data Format

Your input file should be structured as follows:

| Person | Item_1 | Item_2 | Item_3 | ... |
|--------|--------|--------|--------|-----|
| Person_1 | 1 | 0 | 1 | ... |
| Person_2 | 1 | 1 | 0 | ... |
| Person_3 | 0 | 1 | 1 | ... |

- **Rows**: Individual persons/respondents
- **Columns**: Test items
- **Values**: 
  - `1` = Correct response
  - `0` = Incorrect response
  - Missing values are supported

A sample dataset (`sample_data.csv`) is included for testing.

## Technical Details

### 2PL Model
The probability of a correct response is modeled as:

```
P(X=1|θ, a, b) = 1 / (1 + exp(-a(θ - b)))
```

Where:
- `θ` (theta) = Person ability
- `a` = Item discrimination (slope)
- `b` = Item difficulty (location)

### EM Algorithm

**E-Step**: Calculate expected posterior probabilities for each person at each quadrature point

**M-Step**: Optimize item parameters by maximizing the marginal log-likelihood using L-BFGS-B

**Convergence**: Monitored by maximum parameter change across iterations

### EAP Scoring

Person abilities are estimated as the expected value of the posterior distribution:

```
θ_EAP = Σ θ_k * P(θ_k|X)
```

Standard errors are calculated as the standard deviation of the posterior.

## Dependencies

- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Optimization and statistical functions
- **matplotlib**: Static plotting
- **plotly**: Interactive visualizations
- **openpyxl**: Excel file support

## Output Files

### Parameter Table
- Item names
- Discrimination (a) with standard errors
- Difficulty (b) with standard errors

### Person Scores
- Person identifiers
- Ability estimates (θ)
- Standard errors
- Raw scores

## Tips for Best Results

1. **Sample Size**: At least 200-300 persons recommended for stable estimates
2. **Item Quality**: Items should have reasonable discrimination (point-biserial > 0.2)
3. **Quadrature Points**: 40 points is usually sufficient; increase for higher precision
4. **Convergence**: Default tolerance (1e-4) works well for most datasets
5. **Missing Data**: The algorithm handles missing responses naturally

## Troubleshooting

**Convergence Issues**:
- Increase maximum iterations
- Relax convergence tolerance
- Check for items with extreme difficulties (all correct/incorrect)

**Extreme Parameters**:
- Review item statistics in Data Preview tab
- Consider removing problematic items
- Check for sufficient sample size

**Performance**:
- Reduce number of quadrature points for faster computation
- Use smaller datasets for initial testing

## Deployment to Streamlit Cloud

To share this app publicly via Streamlit Community Cloud:

1. **Commit to GitHub**: 
   - Initialize a git repository in this folder (if not done yet).
   - Ensure the `.gitignore` excludes the `venv` folder.
   - Commit and push to a new GitHub repository.

2. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in.
   - Click **New app** and authorize GitHub if needed.
   - Select your repository and branch.
   - Set the Main file path to `irt_calibration_app.py`.
   - Click **Deploy!** 
   
Streamlit will automatically read `requirements.txt` to install the dependencies and launch your app.

## Author

Created for IRT parameter calibration in educational and psychological testing.

## License

Free to use for educational and research purposes.
