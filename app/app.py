from flask import Flask, jsonify
import pandas as pd

scores = pd.read_csv('scores/churn_scores.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'service': 'Churn Prediction API',
        'description': 'Serves precomputed customer churn probability scores',
        'endpoints': {
            '/scores': 'All customers ranked by churn probability',
            '/scores/<customer_id>': 'Churn score for a specific customer',
            '/scores/top/<n>': 'Top N customers most at risk',
            '/scores/threshold/<min_prob>': 'Count and IDs of customers above a churn probability threshold'
        }
    })

@app.route('/scores')
def get_all_scores():
    return jsonify(scores.to_dict(orient='records'))

@app.route('/scores/<int:customer_id>')
def get_customer_score(customer_id):
    result = scores[scores['Customer ID'] == customer_id]
    if result.empty:
        return jsonify({'error': f'Customer {customer_id} not found'}), 404
    row = result.iloc[0]
    return jsonify({
        'customer_id': customer_id,
        'churn_probability': row['churn_probability'],
        'risk_level': 'high' if row['churn_probability'] >= 0.7 else 'medium' if row['churn_probability'] >= 0.4 else 'low'
    })
@app.route('/scores/top/<int:n>')
def get_top_at_risk(n):
    return jsonify(scores.head(n).to_dict(orient='records'))

@app.route('/scores/threshold/<float:min_prob>')
def get_above_threshold(min_prob):
    result = scores[scores['churn_probability'] >= min_prob]
    return jsonify({
        'threshold': min_prob,
        'count': len(result),
        'customer_ids': result['Customer ID'].astype(int).tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)