# app.py
from flask import Flask, json, render_template, request, jsonify
from day_trading_agent import MultiSymbolDayTraderAgent  # Import your agent
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer  # Import your indicator fetcher

app = Flask(__name__)
# symbols = ['DUOL', 'SPY', 'TQQQ', 'AMD', 'ASML','ORCL']
# Initialize your agent and indicator fetcher

indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trade', methods=['POST'])
def trade():
    data = request.json
    print("Printing data: " + str(data))
    stock_symbol = data.get('stock_symbol')
    print("Printing stock symbol: " + str(stock_symbol))
    agent = MultiSymbolDayTraderAgent(symbols=stock_symbol)
    try:
        result = agent.run_sequential()  # Call your agent's trade method
        print("Printing result: " + str(result))
        
        # Pretty-print the result
        response = {
            'success': True,
            'result': result
        }
        return app.response_class(
            response=json.dumps(response, indent=4),  # Pretty-print JSON with 4 spaces
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        return app.response_class(
            response=json.dumps(error_response, indent=4),  # Pretty-print error response
            status=500,
            mimetype='application/json'
        )

# @app.route('/indicators', methods=['POST'])
# def fetch_indicators():
#     data = request.json
#     stock_symbol = data.get('stock_symbol')
#     try:
#         indicators = indicator_fetcher.fetch(stock_symbol)  # Call your fetcher
#         return jsonify({'success': True, 'indicators': indicators})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)