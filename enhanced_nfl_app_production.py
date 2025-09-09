#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NFL Prediction App - Production Ready
Advanced sports betting prediction interface with sophisticated modeling
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import nfl_data_py as nfl
import json
import requests
from typing import Dict, Optional

# Import the enhanced model
from enhanced_nfl_model_production import (
    EnhancedNFLPredictor, 
    compute_enhanced_prop_prediction,
    prepare_enhanced_player_context,
    PredictionResult
)

# Page configuration
st.set_page_config(
    page_title="NFL Prediction Elite Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_data(show_spinner=False, ttl=60*30)
def load_weekly_data(season):
    """Load and cache weekly NFL data"""
    return nfl.import_weekly_data([season], downcast=True)

@st.cache_data(show_spinner=False, ttl=60*60)
def load_team_schedules(season):
    """Load team schedules for current season"""
    try:
        return nfl.import_schedules([season])
    except:
        return pd.DataFrame()

def get_weather_data(city: str, date: str) -> Dict:
    """Get weather data for game location (mock implementation)"""
    # In production, integrate with weather API
    mock_weather = {
        'temperature': np.random.randint(32, 85),
        'wind_speed': np.random.randint(0, 20),
        'precipitation': np.random.random() * 0.5,
        'conditions': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'])
    }
    return mock_weather

def get_betting_data(game_id: str) -> Dict:
    """Get betting market data (mock implementation)"""
    # In production, integrate with sportsbook APIs
    mock_betting = {
        'opening_line': np.random.uniform(200, 300),
        'current_line': np.random.uniform(200, 300),
        'sharp_money_percentage': np.random.randint(20, 80),
        'public_money_percentage': np.random.randint(20, 80),
        'volume': np.random.randint(1000, 10000)
    }
    return mock_betting

def create_prediction_chart(result: Dict, line: float):
    """Create interactive prediction visualization"""
    
    draws = result.get('draws', np.array([]))
    if len(draws) == 0:
        return None
    
    # Create histogram
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=draws,
        nbinsx=50,
        name='Prediction Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add vertical lines
    fig.add_vline(
        x=line, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Line: {line}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=result.get('mean', 0), 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Mean: {result.get('mean', 0):.1f}",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title="Prediction Distribution",
        xaxis_title="Projected Value",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig

def create_confidence_gauge(confidence: float):
    """Create confidence gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence %"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_advanced_metrics(metrics: Dict):
    """Display advanced prediction metrics"""
    
    st.subheader("🔬 Advanced Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Volatility Regime",
            value=metrics.get('volatility_regime', 'Unknown'),
            help="Player's recent performance consistency"
        )
        
        st.metric(
            label="Opponent Strength", 
            value=metrics.get('opponent_strength', 'Unknown'),
            help="Opponent's defensive ranking for this stat"
        )
    
    with col2:
        st.metric(
            label="Injury Risk",
            value=metrics.get('injury_risk', 'Unknown'),
            help="Risk level based on injury report and rest"
        )
        
        st.metric(
            label="Weather Impact",
            value=metrics.get('weather_impact', 'None'),
            help="Expected weather effect on performance"
        )
    
    with col3:
        st.metric(
            label="Model Consensus",
            value=metrics.get('model_consensus', 'Unknown'),
            help="Agreement level between different models"
        )
        
        uncertainty_sources = metrics.get('uncertainty_sources', {})
        total_uncertainty = sum(uncertainty_sources.values())
        st.metric(
            label="Total Uncertainty",
            value=f"{total_uncertainty:.1%}",
            help="Combined uncertainty from all sources"
        )

def main():
    """Main application interface"""
    
    # Header
    st.markdown('<div class="main-header">🏈 NFL Prediction Elite Pro</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Advanced AI-Powered Sports Betting Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")
    
    # Season and basic settings
    season = st.sidebar.number_input("Season (YYYY)", min_value=2012, max_value=2030, value=2024, step=1)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    last_n = st.sidebar.slider("Context window (games)", 4, 16, 8, help="Number of recent games to analyze")
    half_life = st.sidebar.slider("Recency half-life", 3, 12, 5, help="How quickly to discount older games")
    kappa = st.sidebar.slider("Prior strength (κ)", 0, 40, 8, help="Strength of position/league priors")
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    family = st.sidebar.selectbox("Distribution Family", 
                                 ["Ensemble", "Negative Binomial", "Poisson", "Lognormal", "Gamma"], 
                                 index=0,
                                 help="Statistical distribution for modeling")
    nsim = st.sidebar.slider("Simulation draws", 1000, 50000, 10000, step=1000)
    confidence_threshold = st.sidebar.slider("Bet threshold", 0.50, 0.70, 0.52, step=0.01,
                                           help="Minimum confidence required to recommend bet")
    
    # Data loading
    with st.spinner("Loading NFL data..."):
        try:
            weekly_data = load_weekly_data(season)
            schedules = load_team_schedules(season)
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            st.stop()
    
    # Team and player selection
    st.header("🎯 Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Team selection
        team_col = 'recent_team' if 'recent_team' in weekly_data.columns else 'team'
        teams = sorted([t for t in weekly_data[team_col].dropna().unique().tolist()])
        team = st.selectbox("Player Team", teams, help="Select the player's current team")
        
        # Player selection
        team_players = weekly_data[weekly_data[team_col] == team]['player_name'].dropna().unique().tolist()
        player = st.selectbox("Player", sorted(team_players), help="Select the player to analyze")
    
    with col2:
        # Opponent selection
        opp_candidates = sorted(weekly_data[weekly_data[team_col] == team]['opponent_team'].dropna().unique().tolist())
        opponent = st.selectbox("Opponent", opp_candidates, help="Select the opposing team")
        
        # Prop category
        categories = [
            "Passing Yards", "Rushing Yards", "Receiving Yards", "Receptions",
            "Pass Attempts", "Completions", "Rush Attempts", 
            "Pass TDs", "Rush TDs", "Rec TDs", "Rush+Rec Yards", "Fantasy Score"
        ]
        category = st.selectbox("Prop Category", categories, help="Select the statistic to predict")
    
    # Betting line
    st.header("💰 Betting Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        line = st.number_input("Betting Line", min_value=0.0, value=250.0, step=0.5, format="%.1f",
                              help="The sportsbook's over/under line")
    
    with col2:
        snaps_target = st.number_input("Projected Snaps (optional)", min_value=0, max_value=100, value=0, step=1,
                                     help="Expected snap count for usage scaling")
    
    with col3:
        fantasy_system = st.selectbox("Fantasy System", ["PrizePicks", "DraftKings", "FanDuel"],
                                    help="Scoring system for Fantasy Score category")
    
    # Environmental factors
    st.header("🌤️ Environmental Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather Conditions")
        use_weather = st.checkbox("Include weather analysis")
        
        if use_weather:
            temperature = st.number_input("Temperature (°F)", value=72, min_value=-10, max_value=120)
            wind_speed = st.number_input("Wind Speed (mph)", value=5, min_value=0, max_value=50)
            precipitation = st.number_input("Precipitation (inches)", value=0.0, min_value=0.0, max_value=2.0, step=0.1)
        else:
            temperature, wind_speed, precipitation = 72, 5, 0.0
    
    with col2:
        st.subheader("Injury Report")
        injury_status = st.selectbox("Injury Status", 
                                   ["Healthy", "Probable", "Questionable", "Doubtful", "Out"])
        days_rest = st.number_input("Days Rest", value=7, min_value=0, max_value=14,
                                  help="Days since last game")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Analysis")
            include_market = st.checkbox("Include betting market analysis")
            
            if include_market:
                opening_line = st.number_input("Opening Line", value=line)
                sharp_money_pct = st.slider("Sharp Money %", 0, 100, 50)
                public_money_pct = st.slider("Public Money %", 0, 100, 50)
        
        with col2:
            st.subheader("Model Ensemble")
            use_all_models = st.checkbox("Use all available models", value=True)
            model_weights_manual = st.checkbox("Manual model weights")
            
            if model_weights_manual:
                bayesian_weight = st.slider("Bayesian Model Weight", 0.0, 1.0, 0.3)
                rf_weight = st.slider("Random Forest Weight", 0.0, 1.0, 0.25)
                gb_weight = st.slider("Gradient Boost Weight", 0.0, 1.0, 0.25)
                original_weight = st.slider("Original Model Weight", 0.0, 1.0, 0.2)
    
    # Prediction button
    st.header("🎲 Generate Prediction")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🚀 Get Enhanced Prediction", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Running advanced analysis..."):
            try:
                # Prepare enhanced context
                weather_data = {
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'precipitation': precipitation
                } if use_weather else {}
                
                injury_data = {
                    'status': injury_status,
                    'days_rest': days_rest
                }
                
                betting_data = {
                    'opening_line': opening_line if include_market else line,
                    'current_line': line,
                    'sharp_money_percentage': sharp_money_pct if include_market else 50,
                    'public_money_percentage': public_money_pct if include_market else 50
                } if include_market else {}
                
                ctx = prepare_enhanced_player_context(
                    weekly_data, player, team, season, last_n,
                    weather=weather_data,
                    injury_report=injury_data,
                    betting_data=betting_data,
                    opponent_team=opponent
                )
                
                # Prediction settings
                settings = {
                    'half_life': half_life,
                    'kappa': kappa,
                    'family': family.lower() if family != 'Ensemble' else 'ensemble',
                    'nsim': nsim,
                    'snaps_target': snaps_target if snaps_target > 0 else None,
                    'fantasy_system': fantasy_system,
                    'weather': weather_data,
                    'injury_report': injury_data,
                    'betting_data': betting_data,
                    'confidence_threshold': confidence_threshold
                }
                
                # Generate prediction
                result = compute_enhanced_prop_prediction(ctx, category, line, settings)
                
                # Display results
                st.success("✅ Prediction Generated Successfully!")
                
                # Main prediction display
                st.header("📊 Prediction Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    decision_color = "🟢" if result['decision'] == "OVER" else "🔴"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{decision_color} {result['decision']}</h3>
                        <p>Recommendation</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    conf_class = "confidence-high" if result['confidence'] > 0.7 else "confidence-medium" if result['confidence'] > 0.6 else "confidence-low"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 class="{conf_class}">{result['confidence']:.1%}</h3>
                        <p>Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    edge_color = "🟢" if result['edge'] > 0 else "🔴"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{edge_color} {result['edge']:+.1f}</h3>
                        <p>Edge vs Line</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{result['mean']:.1f}</h3>
                        <p>Predicted Mean</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed analysis
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Prediction distribution chart
                    chart = create_prediction_chart(result, line)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                with col2:
                    # Confidence gauge
                    gauge = create_confidence_gauge(result['confidence'])
                    st.plotly_chart(gauge, use_container_width=True)
                
                # Advanced metrics
                if 'advanced_metrics' in result:
                    display_advanced_metrics(result['advanced_metrics'])
                
                # Model breakdown
                st.header("🤖 Model Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Weights")
                    if 'model_weights' in result:
                        weights_df = pd.DataFrame(
                            list(result['model_weights'].items()),
                            columns=['Model', 'Weight']
                        )
                        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(weights_df, use_container_width=True)
                
                with col2:
                    st.subheader("Adjustment Factors")
                    if 'explain' in result:
                        explain = result['explain']
                        adjustments = {
                            'Base Prediction': f"{explain.get('base_prediction', 0):.1f}",
                            'Opponent Adj': f"{explain.get('opponent_adj', 1.0):.2f}x",
                            'Injury Adj': f"{explain.get('injury_adj', 1.0):.2f}x",
                            'Weather Adj': f"{explain.get('weather_adj', 1.0):.2f}x",
                            'Pace Adj': f"{explain.get('pace_adj', 1.0):.2f}x",
                            'Final Mean': f"{explain.get('final_mean', 0):.1f}"
                        }
                        
                        for factor, value in adjustments.items():
                            st.metric(factor, value)
                
                # Statistical details
                with st.expander("📈 Statistical Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Standard Deviation", f"{np.std(result.get('draws', [])):.1f}")
                        st.metric("Skewness", f"{pd.Series(result.get('draws', [])).skew():.2f}")
                    
                    with col2:
                        percentiles = np.percentile(result.get('draws', []), [5, 25, 75, 95])
                        st.metric("5th Percentile", f"{percentiles[0]:.1f}")
                        st.metric("95th Percentile", f"{percentiles[3]:.1f}")
                    
                    with col3:
                        st.metric("Interquartile Range", f"{percentiles[2] - percentiles[1]:.1f}")
                        st.metric("Coefficient of Variation", f"{np.std(result.get('draws', [])) / max(result['mean'], 1e-6):.1%}")
                
                # Betting recommendation
                st.header("💡 Betting Recommendation")
                
                if result['confidence'] >= confidence_threshold:
                    if result['edge'] > 0:
                        st.success(f"""
                        ✅ **RECOMMENDED BET: {result['decision']}**
                        
                        - **Confidence**: {result['confidence']:.1%} (above {confidence_threshold:.1%} threshold)
                        - **Expected Edge**: {result['edge']:+.1f} points
                        - **Probability**: {result['p_over']:.1%} chance of going OVER
                        
                        This bet meets your confidence criteria and shows positive expected value.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **MARGINAL BET: {result['decision']}**
                        
                        High confidence ({result['confidence']:.1%}) but negative edge ({result['edge']:+.1f}).
                        Consider if the line is accurate or if there are other factors.
                        """)
                else:
                    st.error(f"""
                    ❌ **NOT RECOMMENDED**
                    
                    - **Confidence**: {result['confidence']:.1%} (below {confidence_threshold:.1%} threshold)
                    - **Edge**: {result['edge']:+.1f} points
                    
                    Insufficient confidence for a recommended bet. Consider waiting for better spots.
                    """)
                
                # Save to history
                prediction_record = {
                    'timestamp': datetime.now().isoformat(),
                    'player': player,
                    'team': team,
                    'opponent': opponent,
                    'category': category,
                    'line': line,
                    'prediction': result['mean'],
                    'decision': result['decision'],
                    'confidence': result['confidence'],
                    'edge': result['edge']
                }
                
                st.session_state.prediction_history.append(prediction_record)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.header("📚 Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display recent predictions
        st.subheader("Recent Predictions")
        display_df = history_df[['timestamp', 'player', 'category', 'line', 'prediction', 'decision', 'confidence', 'edge']].tail(10)
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['edge'] = display_df['edge'].apply(lambda x: f"{x:+.1f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col2:
            total_predictions = len(history_df)
            recommended = len(history_df[history_df['confidence'] >= confidence_threshold])
            st.metric("Recommended Bets", f"{recommended}/{total_predictions}")
        
        with col3:
            avg_edge = history_df['edge'].mean()
            st.metric("Average Edge", f"{avg_edge:+.1f}")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏈 NFL Prediction Elite Pro - Advanced Sports Betting Analysis</p>
        <p>⚠️ Please bet responsibly. This tool is for entertainment and educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Features")
st.sidebar.markdown("""
- **Hierarchical Bayesian Priors**
- **EWMA Recency Weighting** 
- **Multi-Model Ensemble**
- **Opponent Adjustments**
- **Weather Integration**
- **Injury Impact Modeling**
- **Volatility Analysis**
- **Market Efficiency Signals**
""")

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This application uses advanced statistical methods including:
- Empirical Bayes shrinkage
- Exponentially weighted moving averages
- Ensemble distribution modeling
- Monte Carlo simulation
- Regression analysis
- Uncertainty quantification

**Disclaimer**: For educational purposes only. 
Please gamble responsibly.
""")

if __name__ == "__main__":
    main()