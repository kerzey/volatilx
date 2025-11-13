import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends, HTTPException
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
from typing import Any, Dict, List, Optional

# Import user-related components - UPDATED to include get_user_manager
from user import (
    User,
    UserRead,
    UserCreate,
    UserUpdate,
    get_user_db,
    get_user_manager,
    init_db,
    get_current_user_sync,
)
from db import SessionLocal
from billing import (
    create_subscription_checkout_session,
    ensure_customer,
    get_publishable_key,
    sync_plan_catalogue,
    StripeWebhookConfig,
    enqueue_usage,
    handle_checkout_session_completed,
    handle_invoice_paid,
    handle_subscription_deleted,
    handle_subscription_updated,
    parse_event,
)
from models import SubscriptionPlan, UserSubscription
from sqlalchemy.orm import Session, joinedload

# FastAPI Users imports
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, AuthenticationBackend
from fastapi_users.authentication.strategy import JWTStrategy
from fastapi_users.password import PasswordHelper

# Backend imports
from day_trading_agent import MultiSymbolDayTraderAgent
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer
import json
import urllib.parse

#OpenAI imports
# from ai_agents.openai_service import OpenAIAnalysisService
import jwt
import hashlib
from ai_agents.openai_service import openai_service
from ai_agents.principal_agent import PrincipalAgent
from ai_agents.price_action import PriceActionAnalyzer

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


logger = logging.getLogger(__name__)

ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due"}

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
stripe_webhook_config = (
    StripeWebhookConfig(signing_secret=STRIPE_WEBHOOK_SECRET)
    if STRIPE_WEBHOOK_SECRET
    else None
)


def _build_base_url(request: Request) -> str:
    """Best-effort reconstruction of the public base URL for redirects."""

    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_host:
        scheme = forwarded_proto or request.url.scheme
        return f"{scheme}://{forwarded_host}"

    host = request.headers.get("host") or request.url.netloc
    scheme = forwarded_proto or request.url.scheme
    return f"{scheme}://{host}"


def _serialize_plan(plan: SubscriptionPlan) -> Dict[str, Any]:
    return {
        "id": plan.id,
        "slug": plan.slug,
        "name": plan.name,
        "description": plan.description,
        "monthly_price_cents": plan.monthly_price_cents,
        "monthly_price_dollars": plan.monthly_price_cents / 100,
        "ai_runs_included": plan.ai_runs_included,
        "is_active": plan.is_active,
        "stripe_price_configured": bool(plan.stripe_price_id),
    }


def _serialize_subscription(subscription: Optional[UserSubscription]) -> Optional[Dict[str, Any]]:
    if subscription is None:
        return None

    plan_payload = _serialize_plan(subscription.plan) if subscription.plan else None
    return {
        "id": subscription.id,
        "status": subscription.status,
        "runs_remaining": subscription.runs_remaining,
        "auto_renew": subscription.auto_renew,
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "current_period_start": subscription.current_period_start.isoformat()
        if subscription.current_period_start
        else None,
        "current_period_end": subscription.current_period_end.isoformat()
        if subscription.current_period_end
        else None,
        "plan": plan_payload,
    }


def _find_relevant_subscription(session: Session, user_id: int) -> Optional[UserSubscription]:
    query = (
        session.query(UserSubscription)
        .options(joinedload(UserSubscription.plan))
        .filter(UserSubscription.user_id == user_id)
        .order_by(UserSubscription.created_at.desc())
    )
    subscriptions = query.all()
    for sub in subscriptions:
        if sub.status in ACTIVE_SUBSCRIPTION_STATUSES:
            return sub
    return subscriptions[0] if subscriptions else None


def _consume_subscription_units(
    user_id: int,
    *,
    units: int = 1,
    usage_type: str = "ai_run",
    notes: Optional[str] = None,
) -> int:
    """Deduct subscription allowance for metered AI features.

    Returns the remaining quota after consumption or raises an HTTPException
    if the user is not eligible to run the requested workload.
    """

    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user_id)

        if subscription is None or subscription.plan is None:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "You need an active VolatilX subscription to run premium AI strategies.",
                    "code": "subscription_required",
                    "runs_remaining": 0,
                    "action_label": "View plans",
                    "action_url": "/subscribe?reason=subscription_required",
                },
            )

        if subscription.runs_remaining is None:
            subscription.runs_remaining = 0

        if subscription.runs_remaining < units:
            reset_at = (
                subscription.current_period_end.isoformat()
                if subscription.current_period_end
                else None
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "You’ve used all AI runs included in your plan for this billing cycle.",
                    "code": "quota_exhausted",
                    "runs_remaining": max(subscription.runs_remaining, 0),
                    "action_label": "Manage subscription",
                    "action_url": "/subscribe?reason=quota_exhausted",
                    "renews_at": reset_at,
                },
            )

        subscription.runs_remaining -= units
        enqueue_usage(
            session,
            subscription,
            units=units,
            notes=notes,
            usage_type=usage_type,
        )

        session.commit()
        remaining = subscription.runs_remaining

    return remaining


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        with SessionLocal() as session:
            sync_plan_catalogue(session)
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
            error_detail = urllib.parse.quote_plus(str(e))
            print(f"Database error in Azure callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
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
            error_detail = urllib.parse.quote_plus(str(e))
            print(f"Database error in Google callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
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
price_action_analyzer = PriceActionAnalyzer(analyzer=indicator_fetcher)

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
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    if subscription is None or subscription.plan is None:
        return RedirectResponse(url="/subscribe?from=trade", status_code=status.HTTP_303_SEE_OTHER)

    context = {
        "request": request,
        "user": user,
        "subscription": _serialize_subscription(subscription),
    }
    return templates.TemplateResponse("trade.html", context)

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
        price_action_timeframes_raw = data.get('price_action_timeframes')
        price_action_period_overrides_raw = data.get('price_action_period_overrides')
        # language = data.get('language', 'en')
        
        print("Printing stock symbol:", stock_symbol)
        print("Use AI Analysis:", use_ai_analysis)
        
        if not stock_symbol:
            return JSONResponse(
                content={'success': False, 'error': 'Stock symbol is required'},
                status_code=400
            )
        
        runs_remaining_after: Optional[int] = None
        should_meter = bool(use_ai_analysis or use_principal_agent)
        if should_meter:
            runs_remaining_after = _consume_subscription_units(
                user.id,
                usage_type="trade_analysis",
                notes=f"Trade analysis for {stock_symbol}",
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
        try:
            indicator_fetcher.set_credentials(api_key, secret_key)
        except Exception as cred_error:  # noqa: BLE001 - log but do not fail primary analysis
            logger.warning("Failed to configure shared indicator credentials: %s", cred_error)
        
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

        price_action_timeframes: Optional[List[str]] = None
        if isinstance(price_action_timeframes_raw, str):
            parts = [item.strip() for item in price_action_timeframes_raw.split(',') if item and item.strip()]
            price_action_timeframes = parts or None
        elif isinstance(price_action_timeframes_raw, (list, tuple, set)):
            converted = [str(item).strip() for item in price_action_timeframes_raw if str(item).strip()]
            price_action_timeframes = converted or None

        price_action_period_overrides: Optional[Dict[str, str]] = None
        if isinstance(price_action_period_overrides_raw, dict):
            cleaned_overrides = {
                str(key): str(value)
                for key, value in price_action_period_overrides_raw.items()
                if value is not None and str(value).strip()
            }
            price_action_period_overrides = cleaned_overrides or None

        price_action_analysis: Optional[dict]
        try:
            price_action_analysis = price_action_analyzer.analyze(
                stock_symbol,
                timeframes=price_action_timeframes,
                period_overrides=price_action_period_overrides,
            )
        except Exception as exc:  # noqa: BLE001 - return structured error information
            logger.exception("Price action analysis failed for %s", stock_symbol)
            price_action_analysis = {
                "success": False,
                "symbol": stock_symbol,
                "error": str(exc),
            }

        principal_plan = None
        if use_principal_agent:
            try:
                plan = get_principal_agent_instance().generate_trading_plan(
                    stock_symbol,
                    technical_snapshot=result_data_json,
                    price_action_snapshot=price_action_analysis,
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
            'price_action': price_action_analysis,
            'symbol': stock_symbol,
            'timestamp': str(datetime.now())
        }
        if runs_remaining_after is not None:
            response['runs_remaining'] = runs_remaining_after
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except HTTPException as exc:
        error_payload: Dict[str, Any] = {"success": False}
        detail = exc.detail
        if isinstance(detail, dict):
            error_payload.update(detail)
            if "error" not in error_payload and "message" in error_payload:
                error_payload["error"] = error_payload["message"]
        elif isinstance(detail, str):
            error_payload["error"] = detail
        else:
            error_payload["error"] = "Subscription validation failed."

        if "runs_remaining" not in error_payload:
            error_payload["runs_remaining"] = None

        response = JSONResponse(
            content=error_payload,
            status_code=exc.status_code,
        )
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        return response
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


@app.get("/subscribe", response_class=HTMLResponse)
async def subscribe_page(request: Request, user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    context = {
        "request": request,
        "user": user,
        "current_subscription": _serialize_subscription(subscription) if subscription else None,
    }
    return templates.TemplateResponse("subscription.html", context)

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

##############################################################################################
# Billing Endpoints
##############################################################################################


@app.get("/api/billing/plans")
async def list_subscription_plans(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        plans = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.is_active.is_(True))
            .order_by(SubscriptionPlan.monthly_price_cents.asc())
            .all()
        )
        current_subscription = _find_relevant_subscription(session, user.id)

    payload = {
        "plans": [_serialize_plan(plan) for plan in plans],
        "currency": "usd",
        "current_subscription": _serialize_subscription(current_subscription),
    }
    return JSONResponse(content=payload)


@app.get("/api/billing/subscription")
async def get_current_subscription(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    return JSONResponse(content={"subscription": _serialize_subscription(subscription)})


@app.post("/api/billing/checkout-session")
async def create_checkout_session(request: Request, user: User = Depends(get_current_user_sync)):
    payload = await request.json()
    plan_slug = payload.get("plan_slug")
    if not plan_slug:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "plan_slug is required."},
        )

    with SessionLocal() as session:
        plan = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.slug == plan_slug, SubscriptionPlan.is_active.is_(True))
            .one_or_none()
        )

        if plan is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "Selected plan is unavailable."},
            )

        if not plan.stripe_price_id:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Plan is missing Stripe price configuration."},
            )

        try:
            customer = ensure_customer(
                email=user.email,
                existing_customer_id=user.stripe_customer_id,
            )
        except Exception as exc:  # noqa: BLE001 - want full context in logs
            logger.exception("Failed to ensure Stripe customer for user %s", user.id)
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"detail": "Unable to contact billing provider."},
            )

        if not user.stripe_customer_id:
            db_user = session.query(User).filter(User.id == user.id).one()
            db_user.stripe_customer_id = customer.id
            session.commit()
            user.stripe_customer_id = customer.id

        base_url = _build_base_url(request)
        success_url = f"{base_url}/settings?checkout=success"
        cancel_url = f"{base_url}/settings?checkout=cancel"

        try:
            session_obj = create_subscription_checkout_session(
                price_id=plan.stripe_price_id,
                success_url=success_url,
                cancel_url=cancel_url,
                client_reference_id=str(user.id),
                customer=customer.id,
                metadata={
                    "plan_id": str(plan.id),
                    "plan_slug": plan.slug,
                    "user_id": str(user.id),
                },
            )
        except Exception as exc:  # noqa: BLE001 - propagate message via logs
            logger.exception(
                "Failed to create Stripe checkout session for user %s and plan %s",
                user.id,
                plan.slug,
            )
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"detail": "Unable to initiate checkout session."},
            )

    return JSONResponse(content={"checkout_url": session_obj.url, "session_id": session_obj.id})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(get_current_user_sync)):
    try:
        publishable_key = get_publishable_key()
    except RuntimeError:
        publishable_key = None

    context = {
        "request": request,
        "user": user,
        "stripe_publishable_key": publishable_key,
    }
    return templates.TemplateResponse("settings.html", context)


@app.post("/api/billing/stripe/webhook")
async def stripe_webhook(request: Request):
    if stripe_webhook_config is None:
        raise HTTPException(status_code=503, detail="Stripe webhooks are not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature header.")

    event = parse_event(payload, signature, stripe_webhook_config.signing_secret)
    event_type = event.get("type")

    logger.info("Stripe webhook received: %s", event_type)

    with SessionLocal() as session:
        if event_type in {"customer.subscription.created", "customer.subscription.updated"}:
            handle_subscription_updated(session, event)
        elif event_type == "customer.subscription.deleted":
            handle_subscription_deleted(session, event)
        elif event_type == "invoice.payment_succeeded":
            handle_invoice_paid(session, event)
        elif event_type == "checkout.session.completed":
            handle_checkout_session_completed(session, event)
        else:
            logger.debug("Unhandled Stripe webhook event type: %s", event_type)

    return JSONResponse(content={"received": True})

@app.post("/api/analyze")
async def analyze_trading_data(
    request: Request,
    user: User = Depends(get_current_user_sync)
):
    """Analyze trading data using AI"""
    try:
        runs_remaining_after = _consume_subscription_units(
            user.id,
            usage_type="direct_ai_analysis",
            notes="Direct AI analysis request",
        )
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
            analysis_result["runs_remaining"] = runs_remaining_after
            
            return JSONResponse(
                content=analysis_result,
                status_code=200
            )
        else:
            print(f"AI analysis failed for {symbol}: {analysis_result.get('error', 'Unknown error')}")
            analysis_result["runs_remaining"] = runs_remaining_after
            return JSONResponse(
                content=analysis_result,
                status_code=400
            )
            
    except HTTPException as exc:
        error_payload: Dict[str, Any] = {"success": False}
        detail = exc.detail
        if isinstance(detail, dict):
            error_payload.update(detail)
            if "error" not in error_payload and "message" in error_payload:
                error_payload["error"] = error_payload["message"]
        elif isinstance(detail, str):
            error_payload["error"] = detail
        else:
            error_payload["error"] = "Subscription validation failed."

        if "runs_remaining" not in error_payload:
            error_payload["runs_remaining"] = None

        response = JSONResponse(
            content=error_payload,
            status_code=exc.status_code,
        )
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        return response
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "timestamp": str(datetime.now()),
                "runs_remaining": runs_remaining_after if 'runs_remaining_after' in locals() else None,
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