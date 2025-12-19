# tenant_risk_api.py
# Flask API for Tenant Risk Scoring Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from your website

# Load trained model
with open('tenant_risk_model (1).pkl', 'rb') as f:
    model = pickle.load(f)

print("✓ Tenant Risk Model loaded successfully")

@app.route('/')
def home():
    return jsonify({
        'service': 'Tenant Risk Scoring API',
        'version': '1.0',
        'status': 'online',
        'endpoints': {
            'health': '/api/health',
            'score_tenant': '/api/score-tenant (POST)'
        }
    })

@app.route('/api/score-tenant', methods=['POST'])
def score_tenant():
    """Score tenant risk and compute advance amount"""
    try:
        data = request.json
        
        # Extract features in correct order
        # Map website field names to API expectations
        age = int(data.get('age', 30))
        monthly_income = float(data.get('income', 50000))
        monthly_rent = float(data.get('rent', 20000))
        credit_score = int(data.get('credit_score', 650))
        employment_years = float(data.get('employment_years', 1))
        employment_months = employment_years * 12
        previous_evictions = int(data.get('previous_evictions', 0))
        late_payments = int(data.get('late_payments', 0))
        past_delays_count = float(previous_evictions + late_payments)
        
        # Use reasonable defaults for fields not in website form
        bank_balance_avg = monthly_income * 2  # Estimate
        income_volatility = 0.2  # Default assumption
        # Create feature array
            monthly_income,
            monthly_rent,
            employment_months,
            past_delays_count,
            bank_balance_avg,
            income_volatility,
            geo_risk_score,
            rent_to_income_ratio
        ]])
        
        # Predict probability of default
        pd_proba = model.predict_proba(features)[0][1]
        risk_score = (1 - pd_proba) * 100
        
        # Compute risk-adjusted advance
        months = int(data.get('months', 12))
        total_rent = monthly_rent * months
        
        # Expected loss calculation
        lgd = 0.40  # Loss given default
        el_amount = pd_proba * lgd * total_rent
        el_share = el_amount / total_rent
        
        # Advance formula with risk-based pricing
        required_yield = 0.08
        cost_of_capital = 0.05
        operating_margin = 0.02
        
        total_cost = required_yield + el_share + cost_of_capital + operating_margin
        advance_pct = max(0.0, min(1.0 - total_cost, 0.85))
        
        # Apply risk-based caps
        if pd_proba < 0.05:  # Low risk (Tier A)
            advance_pct = min(advance_pct, 0.80)
            risk_tier = 'A (Low)'
        elif pd_proba < 0.15:  # Medium risk (Tier B)
            advance_pct = min(advance_pct, 0.70)
            risk_tier = 'B (Medium)'
        else:  # High risk (Tier C)
            advance_pct = min(advance_pct, 0.60)
            risk_tier = 'C (High)'
        
        advance_amount = total_rent * advance_pct
        
        # Rewards calculation (demo)
        points = int(total_rent / 100)
        
        return jsonify({
            'default_probability': float(pd_proba),
            'risk_category': risk_tier,
            'recommended_advance_percentage': int(advance_pct * 100),
            'confidence': 'high' if 0.05 < pd_proba < 0.25 else 'medium'
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'tenant_risk_v1'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Tenant Risk API Server Starting...")
    print("="*60)
    print("\nEndpoint: http://localhost:5000/api/score-tenant")
    print("Method: POST")
    print("\nReady to score tenants!\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
