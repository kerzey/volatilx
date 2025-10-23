import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware

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

# Initialize DB tables if not already done (run once!)
init_db()

app = FastAPI()
load_dotenv()
templates = Jinja2Templates(directory="templates")
# Mount the static files directory right after initializing FastAPI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Session middleware for OAuth
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-for-dev")
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
    redirect_uri = request.url_for("google_callback")
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
        data = await request.json()
        print("Printing data:", data)
        stock_symbol = data.get('stock_symbol')
        print("Printing stock symbol:", stock_symbol)
        
        # Initialize trading agent
        agent = MultiSymbolDayTraderAgent(
            symbols=stock_symbol,
            timeframes=['1m','5m', '15m', '30m', '1h', '1d', '1wk']
        )
        
        # Set Alpaca API credentials
        api_key = "PKYJLOK4LZBY56NZKXZLNSG665"#os.getenv("ALPACA_API_KEY"),
        secret_key ="4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1" #os.getenv("ALPACA_SECRET_KEY"),
        agent.set_credentials(api_key=api_key, secret_key=secret_key)
        
        # Set trading parameters
        agent.set_trading_parameters(
            buy_threshold=70,
            sell_threshold=70,
            high_confidence_only=True
        )
        
        # Run the trading analysis
        result = agent.run_sequential()
        result = agent.export_results(result)[0]
        print("Printing result:", result)
        
        response = {
            'success': True,
            'result': result
        }
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"Error in trade endpoint: {e}")
        response = {
            'success': False,
            'error': str(e)
        }
        return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, Form, status, Depends
# from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from authlib.integrations.starlette_client import OAuth
# from starlette.middleware.sessions import SessionMiddleware

# # Import user-related components - UPDATED to include get_user_manager
# from user import User, UserRead, UserCreate, UserUpdate, get_user_db, get_user_manager, init_db, get_current_user_sync
# from db import SessionLocal

# # FastAPI Users imports
# from fastapi_users import FastAPIUsers
# from fastapi_users.authentication import CookieTransport, AuthenticationBackend
# from fastapi_users.authentication.strategy import JWTStrategy
# from fastapi_users.password import PasswordHelper

# # Backend imports
# from day_trading_agent import MultiSymbolDayTraderAgent
# from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer
# import json

# # Initialize DB tables if not already done (run once!)
# init_db()

# app = FastAPI()
# load_dotenv()
# templates = Jinja2Templates(directory="templates")
# # Mount the static files directory right after initializing FastAPI
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Session middleware for OAuth
# app.add_middleware(
#     SessionMiddleware,
#     secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-for-dev")
# )

# ##############################################################################################
# # FastAPI Users Setup - CORRECTED VERSION
# ##############################################################################################
# # JWT Strategy setup
# SECRET = os.getenv("JWT_SECRET", "SECRET")

# def get_jwt_strategy() -> JWTStrategy:
#     return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

# # Cookie transport setup
# cookie_transport = CookieTransport(cookie_name="volatilx_cookie", cookie_max_age=3600)

# # Auth backend setup
# auth_backend = AuthenticationBackend(
#     name="jwt",
#     transport=cookie_transport,
#     get_strategy=get_jwt_strategy,
# )

# # FastAPI Users setup - FIXED: Now uses get_user_manager instead of get_user_db
# fastapi_users = FastAPIUsers[User, int](get_user_manager, [auth_backend])

# # Get current user dependency
# current_active_user = fastapi_users.current_user(active=True)

# # Password helper
# password_helper = PasswordHelper()

# ##############################################################################################
# # OAuth Setup - Azure and Google
# ##############################################################################################
# # OAuth client setup
# oauth = OAuth()
# # Azure OAuth setup
# oauth.register(
#     name="azure",
#     client_id=os.getenv("YOUR_ENTRA_CLIENT_ID"),
#     client_secret=os.getenv("YOUR_ENTRA_CLIENT_SECRET"),
#     server_metadata_url="https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
#     client_kwargs={"scope": "openid email profile"},
# )

# # Google OAuth setup
# oauth.register(
#     name='google',
#     client_id=os.getenv("GOOGLE_CLIENT_ID"),
#     client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
#     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
#     client_kwargs={"scope": "openid email profile"}
# )

# ##############################################################################################
# # OAuth Routes - Azure
# ##############################################################################################
# @app.get("/auth/azure/login")
# async def azure_login(request: Request):
#     redirect_uri = request.url_for("azure_callback")
#     return await oauth.azure.authorize_redirect(request, redirect_uri)

# @app.get("/auth/azure/callback")
# async def azure_callback(request: Request):
#     try:
#         token = await oauth.azure.authorize_access_token(request)
#         user_info = await oauth.azure.parse_id_token(request, token)
#         email = user_info.get("email")
#         oauth_sub = user_info.get("sub")
        
#         if not email:
#             return RedirectResponse(url="/signin?error=no_email")
        
#         # Use sync database operations
#         db = SessionLocal()
#         try:
#             # Query user directly with SQLAlchemy
#             user = db.query(User).filter(User.email == email).first()
            
#             if user is None:
#                 # Create new user
#                 user = User(
#                     email=email,
#                     hashed_password=password_helper.hash(os.urandom(8).hex()),
#                     is_active=True,
#                     is_superuser=False,
#                     is_verified=True,
#                     oauth_provider="azure",
#                     oauth_id=oauth_sub,
#                 )
#                 db.add(user)
#                 db.commit()
#                 db.refresh(user)
            
#             # Issue JWT
#             jwt_strategy = get_jwt_strategy()
#             access_token = await jwt_strategy.write_token(user)
#             response = RedirectResponse("/trade")
#             response.set_cookie(
#                 key="volatilx_cookie", 
#                 value=access_token,
#                 httponly=True,
#                 secure=False,  # Set to True in production
#                 samesite="lax"
#             )
#             return response
            
#         except Exception as e:
#             db.rollback()
#             print(f"Database error in Azure callback: {e}")
#             return RedirectResponse(url="/signin?error=db_error")
#         finally:
#             db.close()
            
#     except Exception as e:
#         print(f"OAuth error in Azure callback: {e}")
#         return RedirectResponse(url="/signin?error=oauth_failed")

# ##############################################################################################
# # OAuth Routes - Google
# ##############################################################################################
# @app.get("/auth/google/login")
# async def google_login(request: Request):
#     redirect_uri = request.url_for("google_callback")
#     return await oauth.google.authorize_redirect(request, redirect_uri)

# @app.get("/auth/google/callback")
# async def google_callback(request: Request):
#     try:
#         token = await oauth.google.authorize_access_token(request)
        
#         # Extract user info
#         if 'userinfo' in token:
#             userinfo = token['userinfo']
#         else:
#             import jwt
#             id_token = token.get('id_token')
#             userinfo = jwt.decode(id_token, options={"verify_signature": False})
        
#         email = userinfo.get("email")
#         oauth_sub = userinfo.get("sub")
        
#         if not email:
#             print("No email found in OAuth response")
#             return RedirectResponse(url="/signin?error=no_email")
        
#         print(f"Processing OAuth for email: {email}")
        
#         # Use sync database operations
#         db = SessionLocal()
#         try:
#             # Query user directly with SQLAlchemy
#             user = db.query(User).filter(User.email == email).first()
            
#             if user is None:
#                 print(f"Creating new user for {email}")
#                 # Create new user
#                 user = User(
#                     email=email,
#                     hashed_password=password_helper.hash(os.urandom(8).hex()),
#                     is_active=True,
#                     is_superuser=False,
#                     is_verified=True,
#                     oauth_provider="google",
#                     oauth_id=oauth_sub,
#                 )
#                 db.add(user)
#                 db.commit()
#                 db.refresh(user)
#                 print(f"User created successfully with ID: {user.id}")
#             else:
#                 print(f"Existing user found: {user.id}")
            
#             # Create JWT token
#             jwt_strategy = get_jwt_strategy()
#             access_token = await jwt_strategy.write_token(user)
            
#             response = RedirectResponse(url="/trade")
#             response.set_cookie(
#                 key="volatilx_cookie", 
#                 value=access_token, 
#                 httponly=True,
#                 secure=False,
#                 samesite="lax",
#                 max_age=3600,  # Add explicit max_age
#                 path="/"       # Add explicit path
#             )
#             print("OAuth successful, redirecting to /trade")
#             return response
            
#         except Exception as e:
#             db.rollback()
#             print(f"Database error in Google callback: {e}")
#             return RedirectResponse(url="/signin?error=db_error")
#         finally:
#             db.close()
            
#     except Exception as e:
#         print(f"OAuth error in Google callback: {e}")
#         return RedirectResponse(url="/signin?error=oauth_failed")

# ##############################################################################################
# # FastAPI Users routers - Register the authentication endpoints
# ##############################################################################################
# app.include_router(
#     fastapi_users.get_auth_router(auth_backend),
#     prefix="/auth/jwt",
#     tags=["auth"],
# )

# app.include_router(
#     fastapi_users.get_register_router(UserRead, UserCreate),
#     prefix="/auth",
#     tags=["auth"]
# )

# app.include_router(
#     fastapi_users.get_users_router(UserRead, UserUpdate),
#     prefix="/users",
#     tags=["users"]
# )

# ##############################################################################################
# # Signin and Signup Page Routes
# ##############################################################################################
# @app.get("/signin", response_class=HTMLResponse)
# async def signin_page(request: Request):
#     return templates. TemplateResponse("signin.html", {"request": request})

# @app.get("/signup", response_class=HTMLResponse)
# async def signup_page(request: Request):
#     return templates. TemplateResponse("signup.html", {"request": request})

# # Basic demo handlers; customize for real authentication logic!
# @app.post("/signin")
# async def handle_signin(
#     request: Request,
#     email: str = Form(...),
#     password: str = Form(...),
# ):
#     # Demo: Redirect to home; integrate FastAPI Users for actual sign in
#     # You need to implement credential check!
#     response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
#     return response

# @app.post("/signup")
# async def handle_signup(
#     request: Request,
#     email: str = Form(...),
#     password: str = Form(...),
#     confirm_password: str = Form(...),
# ):
#     # Demo: Redirect to sign-in. Implement FastAPI Users 'register' logic here!
#     response = RedirectResponse(url="/signin", status_code=status.HTTP_303_SEE_OTHER)
#     return response

# @app.get("/logout")
# async def logout():
#     response = RedirectResponse(url="/signin")
#     response.delete_cookie("volatilx_cookie")
#     return response

# # Exception handler for unauthorized access
# @app.exception_handler(401)
# async def unauthorized_handler(request: Request, exc):
#     if request.url.path.startswith("/api/") or request.headers.get("content-type") == "application/json":
#         return JSONResponse(
#             status_code=401,
#             content={"detail": "Unauthorized"}
#         )
#     return RedirectResponse(url="/signin")

# ##############################################################################################
# # Agent and Indicator Fetcher Initialization
# ##############################################################################################
# indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()

# ##############################################################################################
# # Main Application Routes
# ##############################################################################################
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return RedirectResponse(url="/signin")

# @app.get("/trade", response_class=HTMLResponse)
# async def trade_page(request: Request, user: User = Depends(get_current_user_sync)):
#     return templates.TemplateResponse("trade.html", {"request": request, "user": user})

# @app.post("/trade")
# async def trade(request: Request, user: User = Depends(get_current_user_sync)):
#     try:
#         data = await request.json()
#         print("Printing data:", data)
#         stock_symbol = data.get('stock_symbol')
#         print("Printing stock symbol:", stock_symbol)
        
#         # Initialize trading agent
#         agent = MultiSymbolDayTraderAgent(
#             symbols=stock_symbol,
#             timeframes=['1m','5m', '15m', '30m', '1h', '1d', '1wk']
#         )
        
#         # Set Alpaca API credentials
#         api_key = "PKYJLOK4LZBY56NZKXZLNSG665"#os.getenv("ALPACA_API_KEY"),
#         secret_key ="4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1" #os.getenv("ALPACA_SECRET_KEY"),
#         agent.set_credentials(api_key=api_key, secret_key=secret_key)
        
#         # Set trading parameters
#         agent.set_trading_parameters(
#             buy_threshold=70,
#             sell_threshold=70,
#             high_confidence_only=True
#         )
        
#         # Run the trading analysis
#         result = agent.run_sequential()
#         result = agent.export_results(result)[0]
#         print("Printing result:", result)
        
#         response = {
#             'success': True,
#             'result': result
#         }
#         return JSONResponse(content=response, status_code=status.HTTP_200_OK)
        
#     except Exception as e:
#         print(f"Error in trade endpoint: {e}")
#         response = {
#             'success': False,
#             'error': str(e)
#         }
#         return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
