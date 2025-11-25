import requests

def test_trade_endpoint(base_url="http://127.0.0.1:8000", stock_symbol="AAPL"):
    url = f"{base_url}/analyze"
    payload = {"stock_symbol": stock_symbol}
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:", response.json())
    except Exception as e:
        print("Test failed with error:", str(e))

if __name__ == "__main__":
    test_trade_endpoint()