import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="IRT Parameter Calibration",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class IRT_Calibrator:
    """
    IRT Model Calibrator using EM Algorithm
    Supports both Rasch (1PL) and 2-Parameter Logistic (2PL) models
    """
    
    def __init__(self, model_type='2PL', n_quadrature=40, tol=1e-4, max_iter=100):
        """
        Initialize the calibrator
        
        Parameters:
        -----------
        model_type : str
            'Rasch' for 1-Parameter Logistic or '2PL' for 2-Parameter Logistic
        n_quadrature : int
            Number of quadrature points for numerical integration
        tol : float
            Convergence tolerance for EM algorithm
        max_iter : int
            Maximum number of EM iterations
        """
        self.model_type = model_type
        self.n_quadrature = n_quadrature
        self.tol = tol
        self.max_iter = max_iter
        
        # Set up Gaussian quadrature points and weights
        self.theta_points, self.weights = self._setup_quadrature()
        
        # Initialize parameters
        self.a_params = None  # Discrimination parameters
        self.b_params = None  # Difficulty parameters
        self.a_se = None      # Standard errors for a
        self.b_se = None      # Standard errors for b
        self.convergence_history = []
        
    def _setup_quadrature(self):
        """
        Set up Gaussian quadrature points and weights
        Uses Gauss-Hermite quadrature adapted for standard normal distribution
        """
        # Create evenly spaced points from -4 to 4
        theta_points = np.linspace(-4, 4, self.n_quadrature)
        
        # Calculate weights based on standard normal density
        weights = norm.pdf(theta_points)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        return theta_points, weights
    
    def _probability(self, theta, a, b):
        """
        Calculate probability of correct response using 2PL model
        
        P(X=1|theta, a, b) = 1 / (1 + exp(-a*(theta - b)))
        
        Parameters:
        -----------
        theta : array-like
            Ability levels
        a : float
            Discrimination parameter
        b : float
            Difficulty parameter
        """
        return 1 / (1 + np.exp(-a * (theta - b)))
    
    def _e_step(self, data):
        """
        E-Step: Calculate expected posterior counts
        
        Parameters:
        -----------
        data : numpy array
            Binary response matrix (n_persons x n_items)
        
        Returns:
        --------
        r_ik : numpy array
            Expected posterior probabilities (n_persons x n_quadrature)
        """
        n_persons, n_items = data.shape
        
        # Initialize posterior probabilities
        r_ik = np.zeros((n_persons, self.n_quadrature))
        
        for i in range(n_persons):
            # Calculate likelihood for each quadrature point
            likelihood = np.ones(self.n_quadrature)
            
            for j in range(n_items):
                if not np.isnan(data[i, j]):
                    p_ij = self._probability(self.theta_points, 
                                            self.a_params[j], 
                                            self.b_params[j])
                    
                    if data[i, j] == 1:
                        likelihood *= p_ij
                    else:
                        likelihood *= (1 - p_ij)
            
            # Calculate posterior (likelihood * prior)
            posterior = likelihood * self.weights
            
            # Normalize
            r_ik[i, :] = posterior / np.sum(posterior)
        
        return r_ik
    
    def _m_step_item(self, item_responses, r_ik, item_idx):
        """
        M-Step for a single item: Optimize a and b parameters (2PL) or just b (Rasch)
        
        Parameters:
        -----------
        item_responses : numpy array
            Binary responses for this item (n_persons,)
        r_ik : numpy array
            Expected posterior probabilities (n_persons x n_quadrature)
        item_idx : int
            Index of the item being optimized
        """
        if self.model_type == 'Rasch':
            # Rasch model: only optimize b, keep a=1
            def neg_log_likelihood(b):
                """Negative marginal log-likelihood for this item"""
                a = 1.0  # Fixed discrimination for Rasch
                
                # Prevent extreme parameter values
                if b < -4 or b > 4:
                    return 1e10
                
                nll = 0
                for i in range(len(item_responses)):
                    if not np.isnan(item_responses[i]):
                        p_k = self._probability(self.theta_points, a, b)
                        
                        if item_responses[i] == 1:
                            likelihood_k = p_k
                        else:
                            likelihood_k = 1 - p_k
                        
                        # Expected log-likelihood
                        expected_ll = np.sum(r_ik[i, :] * np.log(likelihood_k + 1e-10))
                        nll -= expected_ll
                
                return nll
            
            # Initial value
            initial_b = self.b_params[item_idx]
            
            # Optimize only b
            result = minimize(neg_log_likelihood, 
                             initial_b,
                             method='L-BFGS-B',
                             bounds=[(-4, 4)])
            
            return 1.0, result.x[0]  # Return a=1, optimized b
        
        else:  # 2PL model
            def neg_log_likelihood(params):
                """Negative marginal log-likelihood for this item"""
                a, b = params
                
                # Prevent extreme parameter values
                if a <= 0.1 or a > 5:
                    return 1e10
                if b < -4 or b > 4:
                    return 1e10
                
                nll = 0
                for i in range(len(item_responses)):
                    if not np.isnan(item_responses[i]):
                        p_k = self._probability(self.theta_points, a, b)
                        
                        if item_responses[i] == 1:
                            likelihood_k = p_k
                        else:
                            likelihood_k = 1 - p_k
                        
                        # Expected log-likelihood
                        expected_ll = np.sum(r_ik[i, :] * np.log(likelihood_k + 1e-10))
                        nll -= expected_ll
                
                return nll
            
            # Initial values
            initial_params = [self.a_params[item_idx], self.b_params[item_idx]]
            
            # Optimize
            result = minimize(neg_log_likelihood, 
                             initial_params,
                             method='L-BFGS-B',
                             bounds=[(0.1, 5), (-4, 4)])
            
            return result.x
    
    def _calculate_standard_errors(self, data, r_ik):
        """
        Calculate standard errors for parameters using observed information matrix
        """
        n_items = data.shape[1]
        self.a_se = np.zeros(n_items)
        self.b_se = np.zeros(n_items)
        
        for j in range(n_items):
            item_responses = data[:, j]
            a, b = self.a_params[j], self.b_params[j]
            
            if self.model_type == 'Rasch':
                # For Rasch model, only calculate SE for b
                info_b = 0
                for i in range(len(item_responses)):
                    if not np.isnan(item_responses[i]):
                        p_k = self._probability(self.theta_points, a, b)
                        w_k = p_k * (1 - p_k)
                        info_b += np.sum(r_ik[i, :] * w_k * a**2)
                
                try:
                    self.b_se[j] = np.sqrt(1.0 / info_b)
                    self.a_se[j] = 0  # No SE for fixed parameter
                except:
                    self.b_se[j] = np.nan
                    self.a_se[j] = 0
            else:
                # For 2PL model, calculate SE for both a and b
                info_matrix = np.zeros((2, 2))
                
                for i in range(len(item_responses)):
                    if not np.isnan(item_responses[i]):
                        p_k = self._probability(self.theta_points, a, b)
                        w_k = p_k * (1 - p_k)
                        
                        # Information matrix elements
                        theta_minus_b = self.theta_points - b
                        
                        info_aa = np.sum(r_ik[i, :] * w_k * theta_minus_b**2)
                        info_ab = np.sum(r_ik[i, :] * w_k * theta_minus_b * a)
                        info_bb = np.sum(r_ik[i, :] * w_k * a**2)
                        
                        info_matrix[0, 0] += info_aa
                        info_matrix[0, 1] += info_ab
                        info_matrix[1, 0] += info_ab
                        info_matrix[1, 1] += info_bb
                
                # Standard errors are square root of diagonal of inverse information matrix
                try:
                    cov_matrix = np.linalg.inv(info_matrix)
                    self.a_se[j] = np.sqrt(max(0, cov_matrix[0, 0]))
                    self.b_se[j] = np.sqrt(max(0, cov_matrix[1, 1]))
                except:
                    self.a_se[j] = np.nan
                    self.b_se[j] = np.nan
    
    def fit(self, data, progress_callback=None):
        """
        Fit the IRT model using EM algorithm
        
        Parameters:
        -----------
        data : numpy array or pandas DataFrame
            Binary response matrix (n_persons x n_items)
        progress_callback : callable, optional
            Function to call with progress updates
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        n_persons, n_items = data.shape
        
        # Initialize parameters
        # Start with reasonable values: a=1 (fixed for Rasch), b=logit(p) where p is proportion correct
        if self.model_type == 'Rasch':
            self.a_params = np.ones(n_items)  # Fixed at 1 for Rasch
        else:
            self.a_params = np.ones(n_items)  # Will be optimized for 2PL
        self.b_params = np.zeros(n_items)
        
        for j in range(n_items):
            p_correct = np.nanmean(data[:, j])
            p_correct = np.clip(p_correct, 0.01, 0.99)  # Avoid extreme values
            self.b_params[j] = -np.log(p_correct / (1 - p_correct))
        
        self.convergence_history = []
        
        # EM Algorithm
        for iteration in range(self.max_iter):
            # E-Step
            r_ik = self._e_step(data)
            
            # M-Step: Update each item's parameters
            old_params = np.concatenate([self.a_params, self.b_params])
            
            for j in range(n_items):
                item_responses = data[:, j]
                new_params = self._m_step_item(item_responses, r_ik, j)
                self.a_params[j], self.b_params[j] = new_params
            
            # Check convergence
            if self.model_type == 'Rasch':
                # For Rasch, only check b parameter changes
                new_params = self.b_params
                param_change = np.max(np.abs(new_params - old_params[n_items:]))
            else:
                # For 2PL, check both a and b
                new_params = np.concatenate([self.a_params, self.b_params])
                param_change = np.max(np.abs(new_params - old_params))
            self.convergence_history.append(param_change)
            
            if progress_callback:
                progress_callback(iteration + 1, self.max_iter, param_change)
            
            if param_change < self.tol:
                st.success(f"✅ Converged after {iteration + 1} iterations!")
                break
        else:
            st.warning(f"⚠️ Maximum iterations ({self.max_iter}) reached without full convergence.")
        
        # Calculate standard errors
        r_ik = self._e_step(data)
        self._calculate_standard_errors(data, r_ik)
        
        return self
    
    def score_persons_eap(self, data):
        """
        Calculate person ability estimates using EAP (Expected A Posteriori)
        
        Parameters:
        -----------
        data : numpy array
            Binary response matrix (n_persons x n_items)
        
        Returns:
        --------
        theta_eap : numpy array
            EAP ability estimates for each person
        theta_se : numpy array
            Standard errors for ability estimates
        """
        n_persons = data.shape[0]
        theta_eap = np.zeros(n_persons)
        theta_se = np.zeros(n_persons)
        
        for i in range(n_persons):
            # Calculate likelihood for each quadrature point
            likelihood = np.ones(self.n_quadrature)
            
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    p_ij = self._probability(self.theta_points, 
                                            self.a_params[j], 
                                            self.b_params[j])
                    
                    if data[i, j] == 1:
                        likelihood *= p_ij
                    else:
                        likelihood *= (1 - p_ij)
            
            # Calculate posterior
            posterior = likelihood * self.weights
            posterior = posterior / np.sum(posterior)
            
            # EAP estimate (expected value)
            theta_eap[i] = np.sum(self.theta_points * posterior)
            
            # Standard error (standard deviation of posterior)
            theta_se[i] = np.sqrt(np.sum((self.theta_points - theta_eap[i])**2 * posterior))
        
        return theta_eap, theta_se
    
    def plot_icc(self, item_idx, item_name=None):
        """
        Plot Item Characteristic Curve for a specific item
        """
        theta_range = np.linspace(-4, 4, 100)
        prob = self._probability(theta_range, 
                                 self.a_params[item_idx], 
                                 self.b_params[item_idx])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=theta_range,
            y=prob,
            mode='lines',
            name=f'Item {item_name or item_idx + 1}',
            line=dict(color='#667eea', width=3)
        ))
        
        # Add vertical line at difficulty parameter
        fig.add_vline(
            x=self.b_params[item_idx],
            line_dash="dash",
            line_color="red",
            annotation_text=f"b = {self.b_params[item_idx]:.2f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=f"Item Characteristic Curve - {item_name or f'Item {item_idx + 1}'}",
            xaxis_title="Ability (θ)",
            yaxis_title="P(Correct Response)",
            template="plotly_white",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_all_icc(self, item_names=None):
        """
        Plot all Item Characteristic Curves on one graph
        """
        theta_range = np.linspace(-4, 4, 100)
        
        fig = go.Figure()
        
        for j in range(len(self.a_params)):
            prob = self._probability(theta_range, self.a_params[j], self.b_params[j])
            
            fig.add_trace(go.Scatter(
                x=theta_range,
                y=prob,
                mode='lines',
                name=item_names[j] if item_names else f'Item {j + 1}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="All Item Characteristic Curves",
            xaxis_title="Ability (θ)",
            yaxis_title="P(Correct Response)",
            template="plotly_white",
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def plot_test_information(self):
        """
        Plot Test Information Function
        """
        theta_range = np.linspace(-4, 4, 100)
        information = np.zeros_like(theta_range)
        
        for j in range(len(self.a_params)):
            p = self._probability(theta_range, self.a_params[j], self.b_params[j])
            # Item information: I(θ) = a² * P(θ) * (1 - P(θ))
            item_info = self.a_params[j]**2 * p * (1 - p)
            information += item_info
        
        # Calculate standard error: SE(θ) = 1 / sqrt(I(θ))
        se = 1 / np.sqrt(information + 1e-10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=theta_range,
            y=information,
            mode='lines',
            name='Test Information',
            line=dict(color='#667eea', width=3),
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=theta_range,
            y=se,
            mode='lines',
            name='Standard Error',
            line=dict(color='#f093fb', width=3, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Test Information Function",
            xaxis_title="Ability (θ)",
            yaxis=dict(
                title="Information",
                title_font=dict(color="#667eea"),
                tickfont=dict(color="#667eea")
            ),
            yaxis2=dict(
                title="Standard Error",
                title_font=dict(color="#f093fb"),
                tickfont=dict(color="#f093fb"),
                overlaying='y',
                side='right'
            ),
            template="plotly_white",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def get_parameters_df(self, item_names=None):
        """
        Get calibrated parameters as a DataFrame
        """
        if item_names is None:
            item_names = [f"Item_{i+1}" for i in range(len(self.a_params))]
        
        if self.model_type == 'Rasch':
            # For Rasch, don't show discrimination parameters (all = 1)
            df = pd.DataFrame({
                'Item': item_names,
                'Difficulty (b)': self.b_params,
                'b SE': self.b_se
            })
        else:
            # For 2PL, show both parameters
            df = pd.DataFrame({
                'Item': item_names,
                'Discrimination (a)': self.a_params,
                'a SE': self.a_se,
                'Difficulty (b)': self.b_params,
                'b SE': self.b_se
            })
        
        return df


def main():
    """
    Main Streamlit application
    """
    
    # Header
    st.markdown('<p class="main-header">📊 IRT Parameter Calibration</p>', unsafe_allow_html=True)
    
    # Sidebar settings
    st.sidebar.header("⚙️ Model Settings")
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select IRT Model",
        options=['Rasch (1PL)', '2PL'],
        index=1,
        help="Rasch: All items have equal discrimination (a=1). 2PL: Items have varying discrimination."
    )
    
    # Update subtitle based on model
    if model_type == 'Rasch (1PL)':
        st.markdown('<p class="sub-header">Rasch (1-Parameter Logistic) Model with EM Algorithm</p>', unsafe_allow_html=True)
        model_key = 'Rasch'
    else:
        st.markdown('<p class="sub-header">2-Parameter Logistic Model with EM Algorithm</p>', unsafe_allow_html=True)
        model_key = '2PL'
    
    st.sidebar.markdown("---")
    
    n_quadrature = st.sidebar.slider(
        "Number of Quadrature Points",
        min_value=10,
        max_value=80,
        value=40,
        step=5,
        help="More points = more accurate but slower"
    )
    
    convergence_tol = st.sidebar.select_slider(
        "Convergence Tolerance",
        options=[1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}"
    )
    
    max_iterations = st.sidebar.number_input(
        "Maximum Iterations",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    st.sidebar.markdown("### 📖 About")
    if model_key == 'Rasch':
        st.sidebar.info(
            "This application calibrates item parameters using the "
            "**Rasch (1PL)** model with the EM algorithm. "
            "\n\nIn the Rasch model, all items have equal discrimination (a=1), "
            "and only difficulty parameters (b) are estimated."
            "\n\n**Upload your data** to get started!"
        )
    else:
        st.sidebar.info(
            "This application calibrates item parameters using the "
            "**2-Parameter Logistic (2PL)** model with the EM algorithm. "
            "\n\nThe 2PL model estimates both discrimination (a) and difficulty (b) "
            "parameters for each item."
            "\n\n**Upload your data** to get started!"
        )
    
    # File upload
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your response matrix (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Rows = persons, Columns = items, Values = 0 (incorrect) or 1 (correct)"
        )
    with col2:
        st.write("") # Spacing
        st.write("") # Spacing
        has_id_col = st.checkbox("First column is Person ID", value=True, 
                               help="Uncheck if your data doesn't have a column for person names/IDs and the first column contains item responses.")
        has_header_row = st.checkbox("First row is Item ID", value=True,
                               help="Uncheck if your data doesn't have a header row and the first row contains person responses.")
    
    if uploaded_file is not None:
        # Load data
        try:
            index_col_val = 0 if has_id_col else None
            header_val = 0 if has_header_row else None
            
            if uploaded_file.name.endswith('.csv'):
                data_df = pd.read_csv(uploaded_file, index_col=index_col_val, header=header_val)
            else:
                data_df = pd.read_excel(uploaded_file, index_col=index_col_val, header=header_val)
                
            # Assign default names if missing
            if not has_header_row:
                data_df.columns = [f"Item_{i+1}" for i in range(data_df.shape[1])]
            if not has_id_col:
                data_df.index = [f"Person_{i+1}" for i in range(data_df.shape[0])]
            
            st.success(f"✅ Data loaded: {data_df.shape[0]} persons × {data_df.shape[1]} items")
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🎯 Item Parameters", "📈 Visualizations"])
            
            with tab1:
                st.subheader("Response Matrix Preview")
                st.dataframe(data_df.head(20), use_container_width=True)
                
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Persons", data_df.shape[0])
                
                with col2:
                    st.metric("Total Items", data_df.shape[1])
                
                with col3:
                    mean_score = data_df.mean().mean()
                    st.metric("Mean Item Difficulty", f"{mean_score:.2%}")
                
                # Item statistics
                st.subheader("Item Statistics")
                item_stats = pd.DataFrame({
                    'Item': data_df.columns,
                    'N Responses': data_df.count(),
                    'Proportion Correct': data_df.mean(),
                    'Point-Biserial': [data_df[col].corr(data_df.sum(axis=1)) for col in data_df.columns]
                })
                st.dataframe(item_stats, use_container_width=True)
            
            with tab2:
                st.subheader("Calibrate Item Parameters")
                
                if st.button("🚀 Run Calibration", type="primary"):
                    # Initialize calibrator
                    calibrator = IRT_Calibrator(
                        model_type=model_key,
                        n_quadrature=n_quadrature,
                        tol=convergence_tol,
                        max_iter=max_iterations
                    )
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(iteration, max_iter, change):
                        progress_bar.progress(iteration / max_iter)
                        status_text.text(f"Iteration {iteration}/{max_iter} - Parameter change: {change:.6f}")
                    
                    # Fit model
                    with st.spinner("Calibrating parameters..."):
                        calibrator.fit(data_df, progress_callback=progress_callback)
                    
                    # Store in session state
                    st.session_state['calibrator'] = calibrator
                    st.session_state['data_df'] = data_df
                    
                    progress_bar.empty()
                    status_text.empty()
                
                # Display results if calibration has been run
                if 'calibrator' in st.session_state:
                    calibrator = st.session_state['calibrator']
                    
                    st.subheader("📊 Calibrated Parameters")
                    
                    params_df = calibrator.get_parameters_df(data_df.columns.tolist())
                    
                    # Format based on model type
                    if model_key == 'Rasch':
                        st.dataframe(params_df.style.format({
                            'Difficulty (b)': '{:.3f}',
                            'b SE': '{:.3f}'
                        }), use_container_width=True)
                    else:
                        st.dataframe(params_df.style.format({
                            'Discrimination (a)': '{:.3f}',
                            'a SE': '{:.3f}',
                            'Difficulty (b)': '{:.3f}',
                            'b SE': '{:.3f}'
                        }), use_container_width=True)
                    
                    # Download button
                    csv = params_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Parameters (CSV)",
                        data=csv,
                        file_name="irt_parameters.csv",
                        mime="text/csv"
                    )
                    
                    # Person scoring
                    st.subheader("👥 Person Ability Estimates (EAP)")
                    
                    if st.button("Calculate Person Abilities"):
                        with st.spinner("Calculating EAP estimates..."):
                            theta_eap, theta_se = calibrator.score_persons_eap(data_df.values)
                        
                        person_scores = pd.DataFrame({
                            'Person': data_df.index,
                            'Ability (θ)': theta_eap,
                            'SE': theta_se,
                            'Raw Score': data_df.sum(axis=1)
                        })
                        
                        st.dataframe(person_scores.style.format({
                            'Ability (θ)': '{:.3f}',
                            'SE': '{:.3f}'
                        }), use_container_width=True)
                        
                        # Download person scores
                        csv_scores = person_scores.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Person Scores (CSV)",
                            data=csv_scores,
                            file_name="person_abilities.csv",
                            mime="text/csv"
                        )
            
            with tab3:
                if 'calibrator' in st.session_state:
                    calibrator = st.session_state['calibrator']
                    
                    st.subheader("📈 Item Characteristic Curves")
                    
                    # Option to view individual or all ICCs
                    view_option = st.radio(
                        "Display option:",
                        ["All items", "Individual item"],
                        horizontal=True
                    )
                    
                    if view_option == "All items":
                        fig = calibrator.plot_all_icc(data_df.columns.tolist())
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        selected_item = st.selectbox(
                            "Select item to view:",
                            range(len(data_df.columns)),
                            format_func=lambda x: data_df.columns[x]
                        )
                        
                        fig = calibrator.plot_icc(selected_item, data_df.columns[selected_item])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("ℹ️ Test Information Function")
                    fig_tif = calibrator.plot_test_information()
                    st.plotly_chart(fig_tif, use_container_width=True)
                    
                else:
                    st.info("👆 Please run calibration first in the 'Item Parameters' tab")
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your file has persons as rows and items as columns, with binary values (0/1)")
    
    else:
        # Show example data format
        st.info("👆 Upload a CSV or Excel file to begin")
        
        with st.expander("📝 See example data format"):
            example_data = pd.DataFrame({
                'Item_1': [1, 0, 1, 1, 0],
                'Item_2': [1, 1, 1, 0, 1],
                'Item_3': [0, 0, 1, 1, 0],
                'Item_4': [1, 1, 1, 1, 1],
                'Item_5': [0, 1, 0, 1, 0]
            }, index=['Person_1', 'Person_2', 'Person_3', 'Person_4', 'Person_5'])
            
            st.dataframe(example_data)
            st.caption("Rows = Persons, Columns = Items, Values = 0 (incorrect) or 1 (correct)")


if __name__ == "__main__":
    main()
