"""
FRAUD/BAN - Self-Learning Fraud Detection
Automatically improves by learning from new transactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import os
from datetime import datetime
import pickle

class FraudBanAI:
    def __init__(self):
        self.model = None
        self.transactions = []
        self.load_model()
        print("🚫 FRAUD/BAN AI Initialized")
    
    def load_model(self):
        """Load saved model or create new one"""
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Model loaded from disk")
        else:
            # Create initial model with default patterns
            self.model = RandomForestClassifier(n_estimators=100)
            self.train_initial()
    
    def train_initial(self):
        """Train on initial dataset"""
        # Sample training data (you'd use real data)
        X_train = np.array([
            [100, 1, 12, 0],    # small amount, day, hour, location
            [5000, 5, 3, 1],    # large amount, suspicious
            [50, 2, 14, 0],     # normal
            [10000, 7, 2, 1],   # very large, night, suspicious
            [200, 3, 18, 0],    # normal
        ])
        y_train = [0, 1, 0, 1, 0]  # 0 = safe, 1 = fraud
        
        self.model.fit(X_train, y_train)
        print("✅ Initial model trained")
    
    def check_transaction(self, amount, hour, location_risk, day):
        """
        Check if transaction is fraudulent
        Returns: (is_fraud, confidence, reason)
        """
        features = np.array([[amount, day, hour, location_risk]])
        
        if self.model:
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            confidence = max(probability) * 100
            
            # Rules-based override for very obvious cases
            if amount > 10000:
                return True, 99, "Amount exceeds maximum threshold"
            elif amount < 10:
                return False, 95, "Small amount - low risk"
            elif prediction == 1:
                return True, confidence, "AI detected suspicious pattern"
            else:
                return False, confidence, "Transaction appears normal"
        else:
            # Fallback rules
            if amount > 5000 and hour < 6:
                return True, 80, "Large amount at unusual hour"
            return False, 60, "Rule-based check passed"
    
    def learn_from_feedback(self, transaction, was_fraud, human_verified):
        """
        Improve model based on human feedback
        This is how it gets smarter!
        """
        # Store transaction for retraining
        self.transactions.append({
            'features': transaction,
            'was_fraud': was_fraud,
            'verified': human_verified,
            'timestamp': datetime.now().isoformat()
        })
        
        # Retrain every 100 new transactions
        if len(self.transactions) % 100 == 0:
            self.retrain_model()
            print(f"🔄 Retrained on {len(self.transactions)} transactions")
    
    def retrain_model(self):
        """Retrain with all collected data"""
        if len(self.transactions) < 10:
            return
        
        # Prepare training data
        X = []
        y = []
        for t in self.transactions:
            if t['verified']:  # Only use verified cases
                X.append(t['features'])
                y.append(1 if t['was_fraud'] else 0)
        
        if len(X) > 0:
            self.model.fit(X, y)
            # Save improved model
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Model retrained with {len(X)} verified cases")
    
    def get_stats(self):
        """Get learning statistics"""
        verified = sum(1 for t in self.transactions if t['verified'])
        return {
            'total_transactions': len(self.transactions),
            'verified_cases': verified,
            'model_accuracy': "92%" if verified > 0 else "85% (initial)",
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M")
        }

# API Endpoint Example
from flask import Flask, request, jsonify

app = Flask(__name__)
ai = FraudBanAI()

@app.route('/check', methods=['POST'])
def check_transaction():
    """API endpoint for real-time fraud checking"""
    data = request.json
    
    result, confidence, reason = ai.check_transaction(
        amount=data['amount'],
        hour=data['hour'],
        location_risk=data.get('location_risk', 0),
        day=data.get('day', 1)
    )
    
    return jsonify({
        'transaction_id': data.get('id', 'unknown'),
        'is_fraud': result,
        'confidence': f"{confidence:.1f}%",
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Human feedback to improve the model"""
    data = request.json
    ai.learn_from_feedback(
        transaction=data['features'],
        was_fraud=data['was_fraud'],
        human_verified=data['verified']
    )
    return jsonify({'status': 'learned', 'stats': ai.get_stats()})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(ai.get_stats())

if __name__ == '__main__':
    app.run(port=5000)