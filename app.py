# # app.py
# from flask import Flask, json, render_template, request, jsonify
# from day_trading_agent import MultiSymbolDayTraderAgent  # Import your agent
# from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer  # Import your indicator fetcher

# app = Flask(__name__)
# # symbols = ['DUOL', 'SPY', 'TQQQ', 'AMD', 'ASML','ORCL']
# # Initialize your agent and indicator fetcher

# indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/trade', methods=['POST'])
# def trade():
#     data = request.json
#     print("Printing data: " + str(data))
#     stock_symbol = data.get('stock_symbol')
#     print("Printing stock symbol: " + str(stock_symbol))
#     agent = MultiSymbolDayTraderAgent(symbols=stock_symbol)
#     try:
#         result = agent.run_sequential()  # Call your agent's trade method
#         print("Printing result: " + str(result))
        
#         # Pretty-print the result
#         response = {
#             'success': True,
#             'result': result
#         }
#         return app.response_class(
#             response=json.dumps(response, indent=4),  # Pretty-print JSON with 4 spaces
#             status=200,
#             mimetype='application/json'
#         )
#     except Exception as e:
#         error_response = {
#             'success': False,
#             'error': str(e)
#         }
#         return app.response_class(
#             response=json.dumps(error_response, indent=4),  # Pretty-print error response
#             status=500,
#             mimetype='application/json'
#         )

# # @app.route('/indicators', methods=['POST'])
# # def fetch_indicators():
# #     data = request.json
# #     stock_symbol = data.get('stock_symbol')
# #     try:
# #         indicators = indicator_fetcher.fetch(stock_symbol)  # Call your fetcher
# #         return jsonify({'success': True, 'indicators': indicators})
# #     except Exception as e:
# #         return jsonify({'success': False, 'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from day_trading_agent import MultiSymbolDayTraderAgent
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Mount the static files directory right after initializing FastAPI
app.mount("/static", StaticFiles(directory="static"), name="static")

# If you have static files (css, js), use this:
# app.mount("/static", StaticFiles(directory="static"), name="static")

indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/trade")
async def trade(request: Request):
    try:
        data = await request.json()
        print("Printing data:", data)
        stock_symbol = data.get('stock_symbol')
        print("Printing stock symbol:", stock_symbol)
        agent = MultiSymbolDayTraderAgent(
            symbols=stock_symbol,
            timeframes=['5m', '15m', '30m', '1h', '1d']
            )
        api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
        secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
        agent.set_credentials(api_key=api_key, secret_key=secret_key)
         # Set trading parameters
        agent.set_trading_parameters(
            buy_threshold=70,
            sell_threshold=70,
            high_confidence_only=True
        )
        result = agent.run_sequential()
        result = agent.export_results(result)[0]
        print("Printing result:", result)
        response = {
            'success': True,
            'result': result
        }
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
        }
        return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# @app.post("/indicators")
# async def fetch_indicators(request: Request):
#     try:
#         data = await request.json()
#         stock_symbol = data.get('stock_symbol')
#         indicators = indicator_fetcher.fetch(stock_symbol)
#         return JSONResponse(content={'success': True, 'indicators': indicators})
#     except Exception as e:
#         return JSONResponse(content={'success': False, 'error': str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

#  print("🚀 Multi-Symbol Day Trading Agent")
#     print("=" * 50)
    
#     # Example with single symbol - use valid timeframes only
#     symbols = ['']#,'ASML','TSLA', "LULU","ORCL"]#,'SpY','QQQ','ETH','SMCI','BTC','NVDA']
    
#     # Initialize the agent with valid timeframes (no 90m)
#     multi_trader = MultiSymbolDayTraderAgent(
#         symbols=symbols,
#         timeframes=['5m', '15m', '30m', '1h', '1d']  # Removed 90m, 5d, 1wk for Alpaca compatibility
#     )
#     api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
#     secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
#     multi_trader.set_credentials(api_key=api_key, secret_key=secret_key)
    
#     # Set trading parameters
#     multi_trader.set_trading_parameters(
#         buy_threshold=70,
#         sell_threshold=70,
#         high_confidence_only=True
#     )
    
#     # Print current status
#     multi_trader.print_status()
    
#     # You would uncomment and set your actual Alpaca credentials here:
#     # multi_trader.set_credentials('your_api_key', 'your_secret_key')
    
#     print(f"\n🔍 Running analysis...")
    
#     # Run sequential analysis (recommended for testing)
#     results = multi_trader.run_sequential()
#     # print("Resutls ", results)
#     # Export results to file
#     if results:
#         multi_trader.export_results(results)

