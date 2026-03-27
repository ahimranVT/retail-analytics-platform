from flask import Flask, jsonify
import pandas as pd

scores = pd.read_csv('scores/churn_scores.csv')

app = Flask(__name__)

@app.route('/scores')
def get_all_scores():
    return jsonify(scores.to_dict(orient='records'))

@app.route('/scores/<int:customer_id>')
def get_customer_score(customer_id):

    result = scores[scores['Customer ID'] == customer_id]

    if result.empty:
        return jsonify({'error': 'Customer not found'}), 404
    return jsonify(result.iloc[0].to_dict())

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