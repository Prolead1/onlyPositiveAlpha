"""
Alpha parameter sets per regime.

These parameters are provided by the alpha team. 
Update these with actual values once your teammates provide them.
"""

ALPHA_PARAMS_PER_REGIME = {
    "risk-on": {
        "name": "Risk-On (Aggressive)",
        "description": "High volatility + positive momentum - aggressive positioning",
        
        # Imbalance Alpha adjustments
        "imbalance_threshold": 0.03,      # 3% (tighter threshold = more signals)
        "imbalance_sensitivity": 1.5,     # Amplify signal by 1.5x
        "imbalance_position_size": 1.0,   # Full position size
        
        # Spread Alpha adjustments
        "spread_p50_discount": 0.95,      # 95% of median spread is "tight"
        "spread_p75_discount": 0.90,      # 90% of p75 is "wide"
        "spread_sensitivity": 1.5,        # Amplify signal by 1.5x
        
        # Risk params
        "volatility_target": 20,          # 20% annualized vol
        "leverage": 2.0,                  # 2x leverage
        "max_drawdown_pct": 15,           # 15% max drawdown
        "max_position_concentration": 0.3,# 30% max in single market
        
        # Expected performance
        "expected_sharpe": 1.2,
        "expected_return_pct": 45,        # 45% annual
    },
    
    "consolidation": {
        "name": "Consolidation (Moderate)",
        "description": "Low volatility + sideways - balanced positioning",
        
        # Imbalance Alpha adjustments
        "imbalance_threshold": 0.05,      # 5% (wider threshold = fewer signals)
        "imbalance_sensitivity": 1.0,     # Normal signal
        "imbalance_position_size": 0.7,   # 70% position size
        
        # Spread Alpha adjustments
        "spread_p50_discount": 1.0,       # Use actual median
        "spread_p75_discount": 1.0,       # Use actual p75
        "spread_sensitivity": 1.0,        # Normal signal
        
        # Risk params
        "volatility_target": 12,          # 12% annualized vol
        "leverage": 1.0,                  # 1x (no leverage)
        "max_drawdown_pct": 10,           # 10% max drawdown
        "max_position_concentration": 0.2,# 20% max in single market
        
        # Expected performance
        "expected_sharpe": 0.8,
        "expected_return_pct": 15,        # 15% annual
    },
    
    "risk-off": {
        "name": "Risk-Off (Conservative)",
        "description": "High volatility + negative momentum - defensive positioning",
        
        # Imbalance Alpha adjustments
        "imbalance_threshold": 0.08,      # 8% (very wide threshold = minimal signals)
        "imbalance_sensitivity": 0.5,     # Dampen signal by 0.5x
        "imbalance_position_size": 0.3,   # 30% position size
        
        # Spread Alpha adjustments
        "spread_p50_discount": 1.05,      # 105% of median (more restrictive)
        "spread_p75_discount": 1.10,      # 110% of p75 (more restrictive)
        "spread_sensitivity": 0.5,        # Dampen signal by 0.5x
        
        # Risk params
        "volatility_target": 8,           # 8% annualized vol
        "leverage": 0.5,                  # 0.5x (half leverage / defensive)
        "max_drawdown_pct": 5,            # 5% max drawdown
        "max_position_concentration": 0.1,# 10% max in single market
        
        # Expected performance
        "expected_sharpe": 0.5,
        "expected_return_pct": 5,         # 5% annual
    },
}

def get_alpha_params(regime: str) -> dict:
    """Retrieve alpha parameters for a given regime."""
    return ALPHA_PARAMS_PER_REGIME.get(regime, ALPHA_PARAMS_PER_REGIME["consolidation"])

def get_all_regimes() -> list[str]:
    """Get list of all supported regimes."""
    return list(ALPHA_PARAMS_PER_REGIME.keys())
