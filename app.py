import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.trustedhost import ProxyHeadersMiddleware
# from starlette.middleware.trustedhost import ForwardedMiddleware
from starlette.responses import RedirectResponse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

# Import user-related components - UPDATED to include get_user_manager
from user import User, UserRead, UserCreate, UserUpdate, get_user_db, get_user_manager, init_db, get_current_user_sync
from db import SessionLocal

# FastAPI Users imports
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, AuthenticationBackend
from fastapi_users.authentication.strategy import JWTStrategy
from fastapi_users.password import PasswordHelper

# Backend imports
from day_trading_agent import MultiSymbolDayTraderAgent
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer
import json

#OpenAI imports
# from ai_agents.openai_service import OpenAIAnalysisService
import jwt
import hashlib
from ai_agents.openai_service import openai_service
from ai_agents.principal_agent import PrincipalAgent

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Try to create tables anyway
        from db import Base, engine
        Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Session middleware for OAuth
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-for-dev")
)
# app.add_middleware(ForwardedMiddleware)
# app.add_middleware(ProxyHeadersMiddleware) 

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://volatilx.com",
        "https://www.volatilx.com",
        "https://volatilx.ai",
        "https://www.volatilx.ai",
        "http://127.0.0.1:8000"  # keep this if you use local dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

##############################################################################################
# FastAPI Users Setup - CORRECTED VERSION
##############################################################################################
# JWT Strategy setup
SECRET = os.getenv("JWT_SECRET", "SECRET")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

# Cookie transport setup
cookie_transport = CookieTransport(cookie_name="volatilx_cookie", cookie_max_age=3600)

# Auth backend setup
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# FastAPI Users setup - FIXED: Now uses get_user_manager instead of get_user_db
fastapi_users = FastAPIUsers[User, int](get_user_manager, [auth_backend])

# Get current user dependency
current_active_user = fastapi_users.current_user(active=True)

# Password helper
password_helper = PasswordHelper()

##############################################################################################
# OAuth Setup - Azure and Google
##############################################################################################
# OAuth client setup
oauth = OAuth()
# Azure OAuth setup
oauth.register(
    name="azure",
    client_id=os.getenv("YOUR_ENTRA_CLIENT_ID"),
    client_secret=os.getenv("YOUR_ENTRA_CLIENT_SECRET"),
    server_metadata_url="https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Google OAuth setup
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

##############################################################################################
# OAuth Routes - Azure
##############################################################################################
@app.get("/auth/azure/login")
async def azure_login(request: Request):
    redirect_uri = request.url_for("azure_callback")
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@app.get("/auth/azure/callback")
async def azure_callback(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user_info = await oauth.azure.parse_id_token(request, token)
        email = user_info.get("email")
        oauth_sub = user_info.get("sub")
        
        if not email:
            return RedirectResponse(url="/signin?error=no_email")
        
        # Use sync database operations
        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            
            if user is None:
                # Create new user
                user = User(
                    email=email,
                    hashed_password=password_helper.hash(os.urandom(8).hex()),
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    oauth_provider="azure",
                    oauth_id=oauth_sub,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            
            # Issue JWT
            jwt_strategy = get_jwt_strategy()
            access_token = await jwt_strategy.write_token(user)
            response = RedirectResponse("/trade")
            response.set_cookie(
                key="volatilx_cookie", 
                value=access_token,
                httponly=True,
                secure=False,  # Set to True in production
                samesite="lax"
            )
            return response
            
        except Exception as e:
            db.rollback()
            print(f"Database error in Azure callback: {e}")
            return RedirectResponse(url="/signin?error=db_error")
        finally:
            db.close()
            
    except Exception as e:
        print(f"OAuth error in Azure callback: {e}")
        return RedirectResponse(url="/signin?error=oauth_failed")

##############################################################################################
# OAuth Routes - Google
##############################################################################################
@app.get("/auth/google/login")
async def google_login(request: Request):
    # host = request.headers.get("host", "www.volatilx.com")
    # redirect_uri = f"https://{host}/auth/google/callback"
    # return await oauth.google.authorize_redirect(request, redirect_uri)
    host = request.headers.get("host", "www.volatilx.com")

    is_local = host.startswith(("127.0.0.1", "localhost"))
    scheme = "http" if is_local else "https"

    redirect_uri = f"{scheme}://{host}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        
        # Extract user info
        if 'userinfo' in token:
            userinfo = token['userinfo']
        else:
            import jwt
            id_token = token.get('id_token')
            userinfo = jwt.decode(id_token, options={"verify_signature": False})
        
        email = userinfo.get("email")
        oauth_sub = userinfo.get("sub")
        
        if not email:
            print("No email found in OAuth response")
            return RedirectResponse(url="/signin?error=no_email")
        
        print(f"Processing OAuth for email: {email}")
        
        # Use sync database operations
        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            
            if user is None:
                print(f"Creating new user for {email}")
                # Create new user
                user = User(
                    email=email,
                    hashed_password=password_helper.hash(os.urandom(8).hex()),
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    oauth_provider="google",
                    oauth_id=oauth_sub,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                print(f"User created successfully with ID: {user.id}")
            else:
                print(f"Existing user found: {user.id}")
            
            # Create JWT token
            jwt_strategy = get_jwt_strategy()
            access_token = await jwt_strategy.write_token(user)
            
            response = RedirectResponse(url="/trade")
            response.set_cookie(
                key="volatilx_cookie", 
                value=access_token, 
                httponly=True,
                secure=False,
                samesite="lax",
                max_age=3600,  # Add explicit max_age
                path="/"       # Add explicit path
            )
            print("OAuth successful, redirecting to /trade")
            return response
            
        except Exception as e:
            db.rollback()
            print(f"Database error in Google callback: {e}")
            return RedirectResponse(url="/signin?error=db_error")
        finally:
            db.close()
            
    except Exception as e:
        print(f"OAuth error in Google callback: {e}")
        return RedirectResponse(url="/signin?error=oauth_failed")

##############################################################################################
# FastAPI Users routers - Register the authentication endpoints
##############################################################################################
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"]
)

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

##############################################################################################
# Signin and Signup Page Routes
##############################################################################################
@app.get("/signin", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates. TemplateResponse("signin.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates. TemplateResponse("signup.html", {"request": request})

# Basic demo handlers; customize for real authentication logic!
@app.post("/signin")
async def handle_signin(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    # Demo: Redirect to home; integrate FastAPI Users for actual sign in
    # You need to implement credential check!
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.post("/signup")
async def handle_signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    # Demo: Redirect to sign-in. Implement FastAPI Users 'register' logic here!
    response = RedirectResponse(url="/signin", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/signin")
    response.delete_cookie("volatilx_cookie")
    return response

# Exception handler for unauthorized access
@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    if request.url.path.startswith("/api/") or request.headers.get("content-type") == "application/json":
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"}
        )
    return RedirectResponse(url="/signin")

##############################################################################################
# Agent and Indicator Fetcher Initialization
##############################################################################################
indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()

principal_agent_instance: Optional[PrincipalAgent] = None


def get_principal_agent_instance() -> PrincipalAgent:
    """Initialise and cache the principal trading agent."""

    global principal_agent_instance
    if principal_agent_instance is None:
        try:
            principal_agent_instance = PrincipalAgent()
        except Exception as exc:  # noqa: BLE001 - surface initialization issues to caller
            logger.error("Failed to initialise PrincipalAgent: %s", exc)
            raise
    return principal_agent_instance

##############################################################################################
# Main Application Routes
##############################################################################################
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/signin")

@app.get("/trade", response_class=HTMLResponse)
async def trade_page(request: Request, user: User = Depends(get_current_user_sync)):
    return templates.TemplateResponse("trade.html", {"request": request, "user": user})

@app.post("/trade")
async def trade(request: Request, user: User = Depends(get_current_user_sync)):
    try:
        # Get JSON data from request
        data = await request.json()
        # print("=== TRADE ENDPOINT DEBUG ===")
        print("Printing data:", data)
        
        # Extract variables from the request data
        stock_symbol = data.get('stock_symbol')
        use_ai_analysis = data.get('use_ai_analysis', False)
        use_principal_agent = data.get('use_principal_agent', use_ai_analysis)
        include_principal_raw = bool(data.get('include_principal_raw_results', False))
        # language = data.get('language', 'en')
        
        print("Printing stock symbol:", stock_symbol)
        print("Use AI Analysis:", use_ai_analysis)
        
        if not stock_symbol:
            return JSONResponse(
                content={'success': False, 'error': 'Stock symbol is required'},
                status_code=400
            )
        
        # Initialize trading agent
        agent = MultiSymbolDayTraderAgent(
            symbols=stock_symbol,
            timeframes=['2m','5m', '15m', '30m', '1h', '1d', '1wk','1mo']
        )
        
        # Set API credentials
        api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
        secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
        agent.set_credentials(api_key=api_key, secret_key=secret_key)
        
        # Set trading parameters
        agent.set_trading_parameters(
            buy_threshold=70,
            sell_threshold=70,
            high_confidence_only=True
        )
        
        # Run the trading analysis
        print("=== RUNNING TRADING ANALYSIS ===")
        results = agent.run_sequential()
        result_data = None
        result_data_json = {}
        if results:
            result_data = agent.export_results(results)
            # If result_data[1] is a string, parse it:
            if isinstance(result_data[1], str):
                result_data_json = json.loads(result_data[1])  # ensures you have a dict
            else:
                result_data_json = result_data[1]

        print("Raw result for AI:", result_data)
        # AI Analysis if requested
        ai_analysis = None
        if use_ai_analysis and result_data is not None:
            try:
                print("=== STARTING AI ANALYSIS ===")
                
                # # Structure the data properly for AI analysis
                # structured_data = structure_trading_data_for_ai(result_data, stock_symbol)
                # print("Structured data for AI:", structured_data)
                
                # # Use the centralized OpenAI service with structured data
                # ai_analysis = openai_service.analyze_trading_data(structured_data, stock_symbol)
                # print("AI Analysis completed:", ai_analysis.get("success", False))
                ai_analysis = openai_service.analyze_trading_data(result_data, stock_symbol)
                print("AI Analysis completed:", ai_analysis.get("success", False))
                 # DEBUG: Print the full AI response
                print("=== FULL AI ANALYSIS RESPONSE ===")
                print("Success:", ai_analysis.get("success"))
                print("Analysis keys:", list(ai_analysis.get("analysis", {}).keys()) if ai_analysis.get("analysis") else "No analysis key")
                print("Raw response preview:", ai_analysis.get("raw_response", "")[:200] + "..." if ai_analysis.get("raw_response") else "No raw response")
        
                # Print each section
                if ai_analysis.get("analysis"):
                    for section_name, section_content in ai_analysis["analysis"].items():
                        print(f"{section_name}: {section_content[:100]}..." if section_content else f"{section_name}: EMPTY")

            except Exception as e:
                print(f"AI Analysis failed: {e}")
                ai_analysis = {
                    "success": False,
                    "error": f"AI analysis failed: {str(e)}"
                }
        
        elif use_ai_analysis:
            ai_analysis = {
                "success": False,
                "error": "Trading data was not generated; unable to run AI analysis.",
            }

        principal_plan = None
        if use_principal_agent:
            try:
                plan = get_principal_agent_instance().generate_trading_plan(
                    stock_symbol,
                    include_raw_results=include_principal_raw,
                )
                principal_plan = {
                    "success": True,
                    "data": plan,
                }
            except Exception as exc:  # noqa: BLE001 - surfaced in response for client visibility
                logger.exception("Principal agent analysis failed for %s", stock_symbol)
                principal_plan = {
                    "success": False,
                    "error": str(exc),
                }

        response = {
            'success': True,
            'result': result_data_json,
            'ai_analysis': ai_analysis,
            'principal_plan': principal_plan,
            'symbol': stock_symbol,
            'timestamp': str(datetime.now())
        }
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error in trade endpoint: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
        return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/api/principal/plan")
async def generate_principal_plan(request: Request, user: User = Depends(get_current_user_sync)):
    data = await request.json()
    symbol = data.get("symbol")
    include_raw = bool(data.get("include_raw_results", False))

    if not symbol:
        return JSONResponse(
            content={"success": False, "error": "Symbol is required"},
            status_code=400,
        )

    try:
        plan = get_principal_agent_instance().generate_trading_plan(
            symbol,
            include_raw_results=include_raw,
        )
    except ValueError as exc:
        logger.error("Principal agent rejected request for %s: %s", symbol, exc)
        return JSONResponse(
            content={"success": False, "error": str(exc)},
            status_code=400,
        )
    except Exception as exc:  # noqa: BLE001 - capture unexpected agent failures
        logger.exception("Principal agent failed for %s", symbol)
        return JSONResponse(
            content={
                "success": False,
                "error": "Principal agent failed to generate plan",
                "details": str(exc),
            },
            status_code=500,
        )

    return JSONResponse(
        content={
            "success": True,
            "plan": plan,
        },
        status_code=200,
    )

def structure_trading_data_for_ai(result_data: any, symbol: str) -> dict:
    """Structure the trading agent result data for AI analysis"""
    
    print(f"=== STRUCTURING DATA FOR AI ===")
    print(f"Input type: {type(result_data)}")
    
    # Initialize structured data with defaults
    structured = {
        "symbol": symbol,
        "timestamp": str(datetime.now()),
        "analysis_type": "day_trading_analysis",
        "current_price": "N/A",
        "price_change": "N/A",
        "volume": "N/A",
        "market_sentiment": "N/A",
        "signals": {},
        "recommendations": {},
        "technical_indicators": {},
        "trading_result": result_data,
        "data_source": "MultiSymbolDayTraderAgent"
    }
    
    try:
        if isinstance(result_data, dict):
            print("Processing dictionary result...")
            print("Available keys:", list(result_data.keys()))
            
            # Extract data based on common key patterns
            for key, value in result_data.items():
                key_lower = key.lower()
                print(f"Processing key: {key} = {str(value)[:100]}...")
                
                # Price-related data
                if any(price_key in key_lower for price_key in ['price', 'close', 'last']):
                    if isinstance(value, (int, float)):
                        structured["current_price"] = value
                    elif isinstance(value, dict) and 'close' in value:
                        structured["current_price"] = value['close']
                
                # Change/movement data
                elif any(change_key in key_lower for change_key in ['change', 'move', 'diff']):
                    structured["price_change"] = value
                
                # Volume data
                elif 'volume' in key_lower:
                    structured["volume"] = value
                
                # Signal data
                elif any(signal_key in key_lower for signal_key in ['signal', 'buy', 'sell', 'action']):
                    if isinstance(value, dict):
                        structured["signals"].update(value)
                    else:
                        structured["signals"][key] = value
                
                # Recommendation data
                elif any(rec_key in key_lower for rec_key in ['recommend', 'advice', 'suggest']):
                    if isinstance(value, dict):
                        structured["recommendations"].update(value)
                    else:
                        structured["recommendations"][key] = value
                
                # Technical indicator data
                elif any(tech_key in key_lower for tech_key in ['rsi', 'macd', 'sma', 'ema', 'indicator', 'technical']):
                    if isinstance(value, dict):
                        structured["technical_indicators"].update(value)
                    else:
                        structured["technical_indicators"][key] = value
            
            # Try to extract nested data
            if 'analysis' in result_data:
                analysis_data = result_data['analysis']
                if isinstance(analysis_data, dict):
                    structured["signals"].update(analysis_data.get('signals', {}))
                    structured["recommendations"].update(analysis_data.get('recommendations', {}))
            
            # Look for timeframe-specific data
            for timeframe in ['1m', '5m', '15m', '30m', '1h', '1d']:
                if timeframe in result_data:
                    tf_data = result_data[timeframe]
                    if isinstance(tf_data, dict):
                        structured["technical_indicators"][f"{timeframe}_data"] = tf_data
            
            # Determine market sentiment based on signals
            if structured["signals"]:
                buy_signals = sum(1 for k, v in structured["signals"].items() 
                                if 'buy' in str(v).lower() or 'bullish' in str(v).lower())
                sell_signals = sum(1 for k, v in structured["signals"].items() 
                                 if 'sell' in str(v).lower() or 'bearish' in str(v).lower())
                
                if buy_signals > sell_signals:
                    structured["market_sentiment"] = "Bullish"
                elif sell_signals > buy_signals:
                    structured["market_sentiment"] = "Bearish"
                else:
                    structured["market_sentiment"] = "Neutral"
        
        elif isinstance(result_data, list) and len(result_data) > 0:
            print("Processing list result...")
            # If it's a list, try to process the first item
            first_item = result_data[0]
            if isinstance(first_item, dict):
                return structure_trading_data_for_ai(first_item, symbol)
            else:
                structured["raw_analysis"] = str(result_data)
                structured["market_sentiment"] = "Data processed"
        
        else:
            print(f"Processing {type(result_data)} result...")
            structured["raw_analysis"] = str(result_data)
            structured["market_sentiment"] = "Analysis completed"
    
    except Exception as e:
        print(f"Error structuring data: {e}")
        structured["extraction_error"] = str(e)
        structured["market_sentiment"] = "Error in data processing"
    
    print("=== FINAL STRUCTURED DATA ===")
    print(f"Current Price: {structured['current_price']}")
    print(f"Signals: {len(structured['signals'])} items")
    print(f"Recommendations: {len(structured['recommendations'])} items")
    print(f"Technical Indicators: {len(structured['technical_indicators'])} items")
    print(f"Market Sentiment: {structured['market_sentiment']}")
    
    return structured

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(get_current_user_sync)):
    return templates. TemplateResponse("settings.html", {"request": request, "user": user})

@app.post("/api/analyze")
async def analyze_trading_data(
    request: Request,
    user: User = Depends(get_current_user_sync)
):
    """Analyze trading data using AI"""
    try:
        # Get the request data
        data = await request.json()
        symbol = data.get('symbol', 'UNKNOWN')
        trading_data = data.get('trading_data', {})
        
        print(f"AI analysis requested by {user.email} for {symbol}")
        print(f"Trading data keys: {list(trading_data.keys())}")
        
        # Enhance trading data with metadata
        enhanced_trading_data = {
            **trading_data,
            "analysis_requested_by": user.email,
            "analysis_timestamp": str(datetime.now()),
            "symbol": symbol
        }
        
        # Use the centralized OpenAI service
        analysis_result = openai_service.analyze_trading_data(enhanced_trading_data, symbol)
        
        if analysis_result["success"]:
            print(f"AI analysis completed for {symbol}, tokens used: {analysis_result.get('tokens_used', 0)}")
            
            # Add metadata to response
            analysis_result.update({
                "analyzed_by": user.email,
                "analysis_timestamp": str(datetime.now()),
                "symbol": symbol
            })
            
            return JSONResponse(
                content=analysis_result,
                status_code=200
            )
        else:
            print(f"AI analysis failed for {symbol}: {analysis_result.get('error', 'Unknown error')}")
            return JSONResponse(
                content=analysis_result,
                status_code=400
            )
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "timestamp": str(datetime.now())
            },
            status_code=500
        )
    
# Add a health check endpoint for OpenAI
@app.get("/api/test-openai")
async def test_openai_connection(user: User = Depends(get_current_user_sync)):
    """Test the centralized OpenAI API connection"""
    try:
        result = openai_service.test_connection()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            content={
                "success": False, 
                "error": f"OpenAI service initialization error: {str(e)}",
                "error_type": "service_error"
            },
            status_code=500
        )
# Add OpenAI status endpoint for more detailed info
@app.get("/api/openai/status")
async def openai_detailed_status(user: User = Depends(get_current_user_sync)):
    """Get detailed OpenAI service status"""
    try:
        # Test connection
        connection_result = openai_service.test_connection()
        
        # Get API key status (masked for security)
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_key_status = {
            "configured": bool(api_key),
            "key_preview": f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Invalid key format",
            "key_length": len(api_key)
        }
        
        return JSONResponse(content={
            "connection_test": connection_result,
            "api_key_status": api_key_status,
            "service_version": "OpenAI v1.x",
            "timestamp": str(datetime.now())
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": f"Status check failed: {str(e)}",
                "timestamp": str(datetime.now())
            },
            status_code=500
        )
# if __name__ == "__main__":
#     import uvicorn

#     # Run the FastAPI app with Uvicorn
#     uvicorn.run(
#         "app:app",  # "app" is the filename, and "app" is the FastAPI instance
#         host="127.0.0.1",  # Localhost
#         port=8000,         # Port number
#         reload=True        # Enable auto-reload for development
#     )