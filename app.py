import logging
import os
import time
from functools import partial
import uuid
import re
import smtplib
from email.message import EmailMessage
import html

try:
    from azure.communication.email import EmailClient
except ImportError:  # pragma: no cover - optional dependency
    EmailClient = None  # type: ignore[assignment]
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends, HTTPException, BackgroundTasks
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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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
from symbol_map import (
    SymbolNotFound,
    SUPPORTED_MARKETS,
    get_all_symbol_catalogs,
    get_ticker_map,
    normalize_symbol,
)
from azure_storage import store_ai_report, fetch_reports_for_date

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


logger = logging.getLogger(__name__)

AI_ANALYSIS_TIMEOUT = float(os.getenv("AI_ANALYSIS_TIMEOUT_SECONDS", "45"))

ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due"}
TRIAL_PLAN_SLUG = os.getenv("TRIAL_PLAN_SLUG", "trial")
TRIAL_DURATION_DAYS = int(os.getenv("TRIAL_DURATION_DAYS", "30"))

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
stripe_webhook_config = (
    StripeWebhookConfig(signing_secret=STRIPE_WEBHOOK_SECRET)
    if STRIPE_WEBHOOK_SECRET
    else None
)

# In-memory store for AI analysis results (for demo; use Redis/db for production)
# In-memory store for AI analysis results (for demo; use Redis/db for production)
ai_analysis_jobs = {}

DASHBOARD_ALLOWED_PLAN_SLUGS = {"sigma", "omega"}
DEFAULT_DASHBOARD_REPORT_LIMIT = int(os.getenv("DASHBOARD_MAX_REPORTS", "120"))
CONTACT_EMAIL_RECIPIENT = os.getenv("CONTACT_EMAIL_RECIPIENT")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}
CONTACT_EMAIL_PREFIX = os.getenv("CONTACT_EMAIL_SUBJECT_PREFIX", "VolatilX Contact")
CONTACT_EMAIL_SENDER = os.getenv("CONTACT_EMAIL_SENDER") or SMTP_USERNAME or CONTACT_EMAIL_RECIPIENT
ACS_CONNECTION_STRING = os.getenv("ACS_CONNECTION_STRING")
ACS_EMAIL_SENDER = os.getenv("ACS_EMAIL_SENDER") or CONTACT_EMAIL_SENDER


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


def _get_subscription_for_user(user_id: int) -> Optional[UserSubscription]:
    with SessionLocal() as session:
        return _find_relevant_subscription(session, user_id)


def _check_dashboard_access(user: User) -> Tuple[Optional[UserSubscription], bool]:
    subscription = _get_subscription_for_user(user.id)
    allowed = bool(
        subscription
        and subscription.plan
        and subscription.plan.slug
        and subscription.plan.slug.lower() in DASHBOARD_ALLOWED_PLAN_SLUGS
    )
    return subscription, allowed


def _ensure_trial_subscription(session: Session, user: User) -> None:
    """Provision a complimentary trial subscription when none exists."""

    active_subscription = (
        session.query(UserSubscription)
        .filter(UserSubscription.user_id == user.id)
        .filter(UserSubscription.status.in_(ACTIVE_SUBSCRIPTION_STATUSES))
        .order_by(UserSubscription.created_at.desc())
        .first()
    )
    if active_subscription is not None:
        logger.debug(
            "User %s already has active subscription %s",
            user.email,
            active_subscription.id,
        )
        return

    trial_plan = (
        session.query(SubscriptionPlan)
        .filter(SubscriptionPlan.slug == TRIAL_PLAN_SLUG)
        .one_or_none()
    )
    if trial_plan is None:
        logger.warning(
            "Trial plan '%s' not found; skipping trial provisioning for user %s",
            TRIAL_PLAN_SLUG,
            user.email,
        )
        return

    prior_trial = (
        session.query(UserSubscription)
        .filter(UserSubscription.user_id == user.id)
        .filter(UserSubscription.plan_id == trial_plan.id)
        .first()
    )
    if prior_trial is not None:
        logger.debug("User %s previously used trial plan", user.email)
        return

    now = datetime.now(timezone.utc)
    trial_subscription = UserSubscription(
        user_id=user.id,
        plan_id=trial_plan.id,
        stripe_customer_id=None,
        stripe_subscription_id=None,
        status="trialing",
        current_period_start=now,
        current_period_end=now + timedelta(days=TRIAL_DURATION_DAYS),
        runs_remaining=trial_plan.ai_runs_included or 0,
        auto_renew=False,
        cancel_at_period_end=True,
    )
    session.add(trial_subscription)
    user.tier = trial_plan.slug
    session.commit()
    logger.info(
        "Provisioned trial subscription %s for user %s with %s runs",
        trial_subscription.id,
        user.email,
        trial_subscription.runs_remaining,
    )


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

        period_end = subscription.current_period_end
        now_utc = datetime.now(timezone.utc)
        if period_end is not None:
            if period_end.tzinfo is None:
                period_end = period_end.replace(tzinfo=timezone.utc)
            if period_end < now_utc:
                subscription.status = "expired"
                subscription.runs_remaining = 0
                session.commit()
                is_trial = bool(subscription.plan and subscription.plan.slug == TRIAL_PLAN_SLUG)
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": (
                            "Your trial period has ended. Upgrade to continue using AI analysis."
                            if is_trial
                            else "Your subscription period has ended. Please renew to continue."
                        ),
                        "code": "trial_expired" if is_trial else "subscription_expired",
                        "runs_remaining": 0,
                        "action_label": "Upgrade plan",
                        "action_url": "/subscribe?reason=trial_expired" if is_trial else "/subscribe?reason=subscription_expired",
                    },
                )

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
        logger.info(
            "Consumed %s units (%s) for user %s subscription %s; remaining=%s",
            units,
            usage_type,
            user_id,
            subscription.id,
            remaining,
        )

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
# OAuth Routes - Azure
##############################################################################################
@app.get("/auth/azure/login")
async def azure_login(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Azure login initiated from %s", client_host)
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
        
        client_host = request.client.host if request.client else "unknown"
        logger.info("Azure OAuth callback for %s from %s", email, client_host)
        # Use sync database operations
        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            
            if user is None:
                # Create new user
                logger.info("Creating Azure user record for %s", email)
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
            
            _ensure_trial_subscription(db, user)

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
            logger.exception("Database error in Azure callback for %s", email)
            print(f"Database error in Azure callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
        finally:
            db.close()
            
    except Exception as e:
        logger.exception("Azure OAuth error")
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
    client_host = request.client.host if request.client else "unknown"
    logger.info("Google login initiated from %s", client_host)
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
        client_host = request.client.host if request.client else "unknown"
        logger.info("Google OAuth callback for %s from %s", email, client_host)
        
        # Use sync database operations
        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            
            if user is None:
                print(f"Creating new user for {email}")
                logger.info("Creating Google user record for %s", email)
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
            
            _ensure_trial_subscription(db, user)
            logger.info("Google user %s assigned to tier %s", user.email, user.tier)

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
            logger.exception("Database error in Google callback for %s", email)
            print(f"Database error in Google callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
        finally:
            db.close()
            
    except Exception as e:
        logger.exception("Google OAuth error")
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
    client_host = request.client.host if request.client else "unknown"
    logger.info("Signin page viewed from %s", client_host)
    return templates. TemplateResponse("signin.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Signup page viewed from %s", client_host)
    return templates. TemplateResponse("signup.html", {"request": request})

# Basic demo handlers; customize for real authentication logic!
@app.post("/signin")
async def handle_signin(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Form signin attempt for %s from %s", email, client_host)
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
    client_host = request.client.host if request.client else "unknown"
    logger.info("Form signup attempt for %s from %s", email, client_host)
    # Demo: Redirect to sign-in. Implement FastAPI Users 'register' logic here!
    response = RedirectResponse(url="/signin", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.get("/logout")
async def logout():
    logger.info("User initiated logout")
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
    logger.info("User %s (%s) opened trade page", user.id, user.email)
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    if subscription is None or subscription.plan is None:
        logger.info("Redirecting user %s to subscription page (no active plan)", user.id)
        return RedirectResponse(url="/subscribe?from=trade", status_code=status.HTTP_303_SEE_OTHER)

    context = {
        "request": request,
        "user": user,
        "subscription": _serialize_subscription(subscription),
        "symbol_catalogs": get_all_symbol_catalogs(),
    }
    return templates.TemplateResponse("trade.html", context)

@app.post("/trade")
async def trade(request: Request, background_tasks: BackgroundTasks, user: User = Depends(get_current_user_sync)):
    try:
        # Get JSON data from request
        data = await request.json()
        # print("=== TRADE ENDPOINT DEBUG ===")
        print("Printing data:", data)
        
        # Extract variables from the request data
        raw_stock_symbol = data.get('stock_symbol')
        stock_symbol = raw_stock_symbol
        market_raw = data.get('market')
        market = str(market_raw).strip().lower() if market_raw is not None else 'equity'
        if market not in SUPPORTED_MARKETS:
            logger.debug("Unsupported market '%s' requested; defaulting to equities", market)
            market = 'equity'
        use_ai_analysis = data.get('use_ai_analysis', False)
        use_principal_agent = data.get('use_principal_agent', use_ai_analysis)
        include_principal_raw = bool(data.get('include_principal_raw_results', False))
        price_action_timeframes_raw = data.get('price_action_timeframes')
        price_action_period_overrides_raw = data.get('price_action_period_overrides')
        # language = data.get('language', 'en')

        symbol_message: Optional[str] = None
        try:
            stock_symbol, message = normalize_symbol(stock_symbol, market=market)
            symbol_message = message
        except SymbolNotFound as exc:
            suggestion_company = None
            suggestion_symbol = exc.suggestion
            if suggestion_symbol:
                suggestion_company = get_ticker_map(market).get(suggestion_symbol)
            error_message = f"Unknown symbol '{raw_stock_symbol}'."
            if suggestion_symbol:
                if suggestion_company:
                    error_message += f" Did you mean {suggestion_company} ({suggestion_symbol})?"
                else:
                    error_message += f" Did you mean {suggestion_symbol}?"
            else:
                error_message += " Please choose a supported ticker."
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": error_message,
                    "code": "unknown_symbol",
                    "suggested_symbol": suggestion_symbol,
                    "suggested_company": suggestion_company,
                },
            )
        
        logger.info(
            "User %s requested trade analysis for %s in %s market (ai=%s principal=%s)",
            user.id,
            stock_symbol,
            market,
            use_ai_analysis,
            use_principal_agent,
        )
        
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
            timeframes=['5m', '15m', '30m', '1h', '1d', '1wk','1mo']
        ,
            market=market,
        )
        
        # Set API credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            logger.error("Alpaca credentials missing; cannot perform trade analysis")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Alpaca credentials are not configured.",
                    "code": "alpaca_credentials_missing",
                },
            )

        agent.set_credentials(api_key=api_key, secret_key=secret_key, base_url=base_url)
        try:
            indicator_fetcher.set_credentials(api_key, secret_key, base_url=base_url)
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

        price_action_analysis: Optional[dict] = None
        try:
            price_action_analysis = price_action_analyzer.analyze(
                stock_symbol,
                timeframes=price_action_timeframes,
                period_overrides=price_action_period_overrides,
                market=market,
            )
        except Exception as exc:  # noqa: BLE001 - return structured error information
            logger.exception("Price action analysis failed for %s", stock_symbol)
            price_action_analysis = {
                "success": False,
                "symbol": stock_symbol,
                "error": str(exc),
            }

        # Prepare background job for AI / principal analysis if requested
        ai_job_id = None
        background_analysis_requested = (use_ai_analysis or use_principal_agent) and result_data is not None
        if background_analysis_requested:
            ai_job_id = str(uuid.uuid4())
            ai_analysis_jobs[ai_job_id] = {"status": "pending", "result": None, "user_id": user.id}

            def run_async_analysis(
                job_id: str,
                technical_data: Any,
                technical_snapshot: Dict[str, Any],
                price_action_snapshot: Optional[Dict[str, Any]],
                symbol: str,
                user_id: Any,
                include_ai: bool,
                include_principal: bool,
                include_principal_raw: bool,
            ) -> None:
                logger.info("Background analysis job %s started for %s", job_id, symbol)
                job_result: Dict[str, Any] = {
                    "ai_analysis": None,
                    "principal_plan": None,
                }
                price_value, price_tf, price_timestamp = _extract_latest_price(technical_snapshot, symbol)
                try:
                    if include_ai:
                        try:
                            logger.info("AI analysis started for %s (job %s)", symbol, job_id)
                            analysis = openai_service.analyze_trading_data(technical_data, symbol)
                            if isinstance(analysis, dict) and price_value is not None:
                                analysis["latest_price"] = price_value
                                if price_tf:
                                    analysis["latest_price_timeframe"] = price_tf
                                if price_timestamp:
                                    analysis["latest_price_timestamp"] = price_timestamp
                            job_result["ai_analysis"] = analysis
                            logger.info("AI analysis completed for %s (job %s)", symbol, job_id)
                        except Exception as ai_exc:  # noqa: BLE001 - store failure in job result
                            logger.exception("AI analysis failed for %s", symbol)
                            job_result["ai_analysis"] = {
                                "success": False,
                                "error": str(ai_exc),
                            }

                    if include_principal:
                        try:
                            logger.info("Principal agent started for %s (job %s)", symbol, job_id)
                            plan = get_principal_agent_instance().generate_trading_plan(
                                symbol,
                                technical_snapshot=technical_snapshot,
                                price_action_snapshot=price_action_snapshot,
                                include_raw_results=include_principal_raw,
                            )
                            if isinstance(plan, dict):
                                if price_value is not None:
                                    plan["latest_price"] = price_value
                                if price_tf:
                                    plan["latest_price_timeframe"] = price_tf
                                if price_timestamp:
                                    plan["latest_price_timestamp"] = price_timestamp
                            job_result["principal_plan"] = {
                                "success": True,
                                "data": plan,
                            }
                            logger.info("Principal agent completed for %s (job %s)", symbol, job_id)
                        except Exception as principal_exc:  # noqa: BLE001 - store failure in job result
                            logger.exception("Principal agent failed for %s", symbol)
                            job_result["principal_plan"] = {
                                "success": False,
                                "error": str(principal_exc),
                            }

                    ai_analysis_jobs[job_id] = {
                        "status": "done",
                        "result": job_result,
                        "user_id": user_id,
                    }
                    store_ai_report(
                        symbol,
                        user_id,
                        {
                            "ai_job_id": job_id,
                            "status": "done",
                            "ai_analysis": job_result.get("ai_analysis"),
                            "principal_plan": job_result.get("principal_plan"),
                            "technical_snapshot": technical_snapshot,
                            "price_action_snapshot": price_action_snapshot,
                        },
                    )
                    logger.info("Background analysis job %s finished", job_id)
                except Exception as exc:  # noqa: BLE001 - surface in job status
                    logger.exception("Background analysis job %s failed", job_id)
                    ai_analysis_jobs[job_id] = {
                        "status": "error",
                        "error": str(exc),
                        "user_id": user_id,
                    }
                    store_ai_report(
                        symbol,
                        user_id,
                        {
                            "ai_job_id": job_id,
                            "status": "error",
                            "error": str(exc),
                            "technical_snapshot": technical_snapshot,
                            "price_action_snapshot": price_action_snapshot,
                        },
                    )

            background_tasks.add_task(
                run_async_analysis,
                ai_job_id,
                result_data,
                result_data_json,
                price_action_analysis,
                stock_symbol,
                user.id,
                use_ai_analysis,
                use_principal_agent,
                include_principal_raw,
            )

        # ...existing code...


        response = {
            'success': True,
            'result': result_data_json,
            'price_action': price_action_analysis,
            'symbol': stock_symbol,
            'market': market,
            'timestamp': str(datetime.now()),
            'ai_job_id': ai_job_id,
            'symbol_message': symbol_message,
            'input_symbol': raw_stock_symbol,
        }
        if runs_remaining_after is not None:
            response['runs_remaining'] = runs_remaining_after
        logger.info(
            "Trade analysis completed for %s by user %s (runs_remaining=%s)",
            stock_symbol,
            user.id,
            response.get('runs_remaining'),
        )
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except HTTPException as exc:
        logger.warning(
            "Trade analysis blocked for user %s: %s",
            user.id,
            exc.detail,
        )
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
    except Exception as exc:
        logger.exception("Trade endpoint failed")
        response = {
            'success': False,
            'error': str(exc)
        }
        return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Endpoint to poll for AI analysis result
@app.get("/api/ai-analysis-result/{job_id}")
async def get_ai_analysis_result(job_id: str, user: User = Depends(get_current_user_sync)):
    job = ai_analysis_jobs.get(job_id)
    if not job:
        return JSONResponse(content={"success": False, "error": "Job not found"}, status_code=404)
    # Only allow the user who started the job to see the result
    if job.get("user_id") != user.id:
        return JSONResponse(content={"success": False, "error": "Unauthorized"}, status_code=403)
    if job["status"] == "pending":
        return JSONResponse(content={"success": False, "status": "pending"}, status_code=200)
    if job["status"] == "done":
        return JSONResponse(content={"success": True, "status": "done", "result": job["result"]}, status_code=200)
    if job["status"] == "error":
        return JSONResponse(content={"success": False, "status": "error", "error": job.get("error")}, status_code=200)
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


@app.post("/api/contact")
async def submit_contact_request(request: Request, background_tasks: BackgroundTasks, user: User = Depends(get_current_user_sync)):
    if not (CONTACT_EMAIL_RECIPIENT and CONTACT_EMAIL_SENDER and SMTP_HOST):
        logger.warning("Contact endpoint requested but email configuration is incomplete")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Contact service is not configured. Please try again later."},
        )

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - invalid JSON
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid JSON payload."},
        ) from None

    def _trim(value: Any, *, max_length: int) -> str:
        if value is None:
            return ""
        text_value = str(value).strip()
        if len(text_value) > max_length:
            return text_value[:max_length].strip()
        return text_value

    name = _trim(payload.get("name") or user.email or "Customer", max_length=120)
    email_address = _trim(payload.get("email") or user.email or "", max_length=255)
    message = payload.get("message")
    if message is None:
        message = ""
    else:
        message = str(message).strip()

    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide your name."},
        )

    if not email_address or "@" not in email_address:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide a valid email address."},
        )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please include a brief message."},
        )

    if len(message) > 4000:
        message = message[:4000].rstrip()

    submission_payload: Dict[str, Any] = {
        "name": name,
        "email": email_address,
        "message": message,
        "user_id": user.id,
        "user_email": user.email,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(_send_contact_email, submission_payload)

    logger.info("Queued contact request from user %s", user.id)

    return JSONResponse(
        content={"success": True},
        status_code=status.HTTP_202_ACCEPTED,
    )


@app.get("/subscribe", response_class=HTMLResponse)
async def subscribe_page(request: Request, user: User = Depends(get_current_user_sync)):
    logger.info("User %s (%s) opened subscription page", user.id, user.email)
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    context = {
        "request": request,
        "user": user,
        "current_subscription": _serialize_subscription(subscription) if subscription else None,
    }
    return templates.TemplateResponse("subscription.html", context)

def _resolve_dashboard_date(raw_date: Optional[str]) -> Tuple[datetime, str, str]:
    """Resolve query date into a midnight UTC timestamp and display labels."""

    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if raw_date:
        candidate = str(raw_date).strip()
        if candidate:
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    parsed = datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
                resolved = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
                return resolved, resolved.strftime("%Y-%m-%d"), resolved.strftime("%b %d, %Y")

    return today_utc, today_utc.strftime("%Y-%m-%d"), today_utc.strftime("%b %d, %Y")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    date: Optional[str] = None,
    symbol: Optional[str] = None,
    user: User = Depends(get_current_user_sync),
):
    subscription, allowed = _check_dashboard_access(user)
    if not allowed:
        query = {"reason": "dashboard_locked"}
        if subscription and subscription.plan and subscription.plan.slug:
            query["current_plan"] = subscription.plan.slug
        redirect_url = "/subscribe"
        if query:
            redirect_url = f"/subscribe?{urllib.parse.urlencode(query)}"
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    target_date, selected_date_iso, selected_date_label = _resolve_dashboard_date(date)
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbol_filter = None
    if symbol:
        symbol_filter_candidate = str(symbol).strip().upper()
        if symbol_filter_candidate:
            symbol_filter = symbol_filter_candidate

    raw_reports = fetch_reports_for_date(
        target_date,
        symbol=symbol_filter,
        max_reports=DEFAULT_DASHBOARD_REPORT_LIMIT,
    )

    prepared_reports: List[Dict[str, Any]] = []
    excluded_reports: List[Dict[str, Any]] = []
    for report in raw_reports:
        summary = _summarize_dashboard_report(report)
        if summary:
            prepared_reports.append(summary)
        else:
            excluded_reports.append(report)

    prepared_reports.sort(key=lambda item: item.get("generated_unix") or 0, reverse=True)

    available_symbols = sorted(
        {
            str(report.get("symbol") or "").upper()
            for report in raw_reports
            if report.get("symbol")
        }
    )

    logger.info(
        "User %s loaded dashboard date=%s symbol=%s reports=%s prepared=%s",
        user.id,
        selected_date_iso,
        symbol_filter,
        len(raw_reports),
        len(prepared_reports),
    )

    context = {
        "request": request,
        "user": user,
        "subscription": _serialize_subscription(subscription) if subscription else None,
        "selected_date": selected_date_iso,
        "selected_date_label": selected_date_label,
        "selected_symbol": symbol_filter,
        "selected_symbol_input": symbol or "",
        "today_date": today_iso,
        "reports": prepared_reports,
        "report_count": len(prepared_reports),
        "raw_report_count": len(raw_reports),
        "excluded_report_count": len(excluded_reports),
        "available_symbols": available_symbols,
        "max_reports": DEFAULT_DASHBOARD_REPORT_LIMIT,
    }
    return templates.TemplateResponse("dashboard.html", context)


def _extract_latest_price(snapshot: Any, symbol: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Best-effort extraction of latest price metadata from analysis snapshots."""

    if snapshot is None:
        return (None, None, None)

    material = snapshot
    if isinstance(material, str):
        try:
            material = json.loads(material)
        except (TypeError, ValueError):
            return (None, None, None)

    if not isinstance(material, dict):
        return (None, None, None)

    symbol_candidates = [symbol, symbol.upper(), symbol.lower()]
    symbol_data = None
    for candidate in symbol_candidates:
        if candidate in material:
            symbol_data = material.get(candidate)
            break

    if not isinstance(symbol_data, dict):
        return (None, None, None)

    price = symbol_data.get("latest_price")
    timeframe = symbol_data.get("latest_price_timeframe")
    timestamp = symbol_data.get("latest_price_timestamp")

    price_map = symbol_data.get("price_by_timeframe")
    if price is None and isinstance(price_map, dict):
        preferred_order = ("1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
        for candidate_tf in preferred_order:
            candidate_price = price_map.get(candidate_tf)
            if candidate_price is not None:
                price = candidate_price
                if timeframe is None:
                    timeframe = candidate_tf
                break
        if price is None:
            for candidate_tf, candidate_price in price_map.items():
                if candidate_price is not None:
                    price = candidate_price
                    if timeframe is None:
                        timeframe = candidate_tf
                    break

    try:
        price_value = float(price) if price is not None else None
    except (TypeError, ValueError):
        price_value = None

    if timestamp is None:
        timestamps_map = symbol_data.get("price_timestamps")
        if isinstance(timestamps_map, dict) and timeframe:
            timestamp = timestamps_map.get(timeframe)

    if timestamp is not None:
        timestamp = str(timestamp)

    return (price_value, timeframe, timestamp)


def _send_contact_email(payload: Dict[str, Any]) -> bool:
    if not CONTACT_EMAIL_RECIPIENT:
        logger.warning("Contact email recipient not configured; skipping send")
        return False

    name = payload.get("name") or "Unknown"
    email_address = payload.get("email") or "unknown@example.com"
    message = payload.get("message") or "(no message provided)"
    submitted_at = payload.get("submitted_at") or datetime.now(timezone.utc).isoformat()

    subject = f"{CONTACT_EMAIL_PREFIX} - {name}" if CONTACT_EMAIL_PREFIX else f"Contact Request - {name}"

    text_body = (
        "New contact request from VolatilX dashboard.\n\n"
        f"Name: {name}\n"
        f"Email: {email_address}\n"
        f"Submitted At: {submitted_at}\n"
        "\nMessage:\n"
        f"{message}\n"
    )

    safe_name = html.escape(str(name))
    safe_email = html.escape(str(email_address))
    safe_message = "<br />".join(html.escape(str(message)).splitlines())
    safe_submitted = html.escape(str(submitted_at))

    html_body = f"""
        <html>
            <body>
                <h2>New VolatilX Contact Request</h2>
                <p><strong>Name:</strong> {safe_name}</p>
                <p><strong>Email:</strong> {safe_email}</p>
                <p><strong>Submitted:</strong> {safe_submitted}</p>
                <hr />
                <p>{safe_message or '<em>(no message provided)</em>'}</p>
            </body>
        </html>
    """

    if ACS_CONNECTION_STRING:
        if EmailClient is None:
            logger.warning(
                "Azure Communication Services email library missing. Install 'azure-communication-email' or disable ACS to use SMTP."
            )
        else:
            sender_address = ACS_EMAIL_SENDER or CONTACT_EMAIL_SENDER
            if not sender_address:
                logger.warning("ACS sender address not configured; skipping ACS send")
            else:
                try:
                    client = EmailClient.from_connection_string(ACS_CONNECTION_STRING)
                    email_message: Dict[str, Any] = {
                        "senderAddress": sender_address,
                        "recipients": {"to": [{"address": CONTACT_EMAIL_RECIPIENT}]},
                        "content": {
                            "subject": subject,
                            "plainText": text_body,
                            "html": html_body,
                        },
                    }
                    if email_address:
                        email_message["replyTo"] = [{"address": email_address}]

                    poller = client.begin_send(email_message)
                    result = poller.result()
                    status = getattr(result, "status", None)
                    if status and str(status).lower() not in {"queued", "accepted", "succeeded"}:
                        logger.warning("ACS email send completed with status %s", status)
                    else:
                        logger.info("Contact email queued via ACS for %s", email_address)
                        return True
                except Exception:  # noqa: BLE001 - logging for operational visibility
                    logger.exception("Failed to send contact email via ACS")

    if not CONTACT_EMAIL_SENDER or not SMTP_HOST:
        logger.warning("No working email transport configured (ACS failed and SMTP incomplete)")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = CONTACT_EMAIL_RECIPIENT
    msg["From"] = CONTACT_EMAIL_SENDER
    if email_address:
        msg["Reply-To"] = email_address
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USERNAME and SMTP_PASSWORD:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(msg)
        logger.info("Contact email sent via SMTP for %s", email_address)
        return True
    except Exception:  # noqa: BLE001 - log and surface failure gracefully
        logger.exception("Failed to send contact email via SMTP")
        return False


def _clean_text_fragment(value: Any, *, max_items: Optional[int] = None) -> str:
    """Convert heterogeneous plan fragments into compact plain-text strings."""

    if value is None:
        return ""

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if max_items is not None:
            items = items[:max_items]
        fragments = [frag for frag in (_clean_text_fragment(item) for item in items) if frag]
        return "; ".join(fragments)

    if isinstance(value, dict):
        fragments = []
        for key, val in value.items():
            cleaned_val = _clean_text_fragment(val)
            if not cleaned_val:
                continue
            label = str(key).replace("_", " ").title()
            fragments.append(f"{label}: {cleaned_val}")
        return "; ".join(fragments)

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        try:
            parsed = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_strategy_for_dashboard(strategy: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(strategy, dict):
        return None

    summary = _clean_text_fragment(strategy.get("summary"))
    next_actions_raw = strategy.get("next_actions")
    actions: List[str] = []
    if isinstance(next_actions_raw, (list, tuple, set)):
        for item in next_actions_raw:
            cleaned = _clean_text_fragment(item)
            if cleaned:
                actions.append(cleaned)
            if len(actions) >= 4:
                break

    result: Dict[str, Any] = {}
    if summary:
        result["summary"] = summary
    if actions:
        result["next_actions"] = actions

    confidence = strategy.get("confidence") or strategy.get("strength")
    if confidence:
        result["confidence"] = _clean_text_fragment(confidence)

    return result or None


def _extract_consensus_snapshot(report: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    snapshot_map = report.get("technical_snapshot")
    if not isinstance(snapshot_map, dict):
        return None

    candidates = [symbol, symbol.upper(), symbol.lower()]
    snapshot = None
    for candidate in candidates:
        candidate_snapshot = snapshot_map.get(candidate)
        if candidate_snapshot:
            snapshot = candidate_snapshot
            break

    if not isinstance(snapshot, dict):
        return None

    consensus = snapshot.get("consensus")
    summary: Dict[str, Any] = {
        "status": snapshot.get("status"),
        "timestamp": snapshot.get("timestamp"),
    }

    if isinstance(consensus, dict):
        summary["recommendation"] = consensus.get("overall_recommendation")
        summary["confidence"] = consensus.get("confidence")
        strength = _safe_float(consensus.get("strength"))
        if strength is not None:
            summary["strength"] = strength
        for field in ("buy_signals", "sell_signals", "hold_signals"):
            if field in consensus:
                summary[field] = consensus.get(field)
        reasoning = consensus.get("reasoning")
        if isinstance(reasoning, list):
            cleaned_reasoning: List[str] = []
            for item in reasoning[:3]:
                cleaned_item = _clean_text_fragment(item)
                if cleaned_item:
                    cleaned_reasoning.append(cleaned_item)
            if cleaned_reasoning:
                summary["reasoning"] = cleaned_reasoning

    decisions = snapshot.get("decisions")
    if isinstance(decisions, dict):
        for focus_tf in ("1d", "4h", "1h"):
            details = decisions.get(focus_tf)
            if isinstance(details, dict):
                focus_summary = {
                    "timeframe": focus_tf,
                    "recommendation": details.get("recommendation"),
                    "confidence": details.get("confidence"),
                }
                for field in ("entry_price", "stop_loss", "take_profit", "risk_reward_ratio"):
                    if details.get(field) is not None:
                        focus_summary[field] = details.get(field)
                summary["focus"] = focus_summary
                break

    return summary or None


def _extract_price_details(report: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    price_action = report.get("price_action_snapshot")
    price_info: Dict[str, Any] = {}
    price_action_summary: Dict[str, Any] = {}

    if isinstance(price_action, dict) and price_action.get("success"):
        per_timeframe = price_action.get("per_timeframe")
        if isinstance(per_timeframe, dict):
            for timeframe in ("1d", "4h", "1h", "30m"):
                timeframe_data = per_timeframe.get(timeframe)
                if isinstance(timeframe_data, dict):
                    price_data = timeframe_data.get("price")
                    if isinstance(price_data, dict):
                        change_pct = _safe_float(price_data.get("close_change_pct"))
                        price_info = {
                            "timeframe": timeframe,
                            "close": price_data.get("close"),
                            "change_pct": change_pct,
                            "volume": price_data.get("volume"),
                            "timestamp": price_data.get("timestamp"),
                        }
                        break

        overview = price_action.get("overview")
        if isinstance(overview, dict):
            price_action_summary["trend_alignment"] = overview.get("trend_alignment")
            key_levels = overview.get("key_levels")
            if isinstance(key_levels, list) and key_levels:
                sorted_levels = sorted(
                    (
                        level
                        for level in key_levels
                        if isinstance(level, dict) and level.get("price") is not None
                    ),
                    key=lambda item: abs(_safe_float(item.get("distance_pct")) or 0.0),
                )
                price_action_summary["key_levels"] = sorted_levels[:3]

            patterns = overview.get("recent_patterns")
            if isinstance(patterns, list) and patterns:
                price_action_summary["recent_patterns"] = patterns[:3]

    return (price_info or None, price_action_summary or None)


def _summarize_dashboard_report(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan_wrapper = report.get("principal_plan")
    if not isinstance(plan_wrapper, dict) or not plan_wrapper.get("success"):
        return None

    plan_data = plan_wrapper.get("data")
    if not isinstance(plan_data, dict):
        return None

    symbol = plan_data.get("symbol") or report.get("symbol")
    if not symbol:
        return None
    symbol_clean = str(symbol).upper()

    generated_dt = _parse_iso_datetime(
        plan_data.get("generated")
        or plan_data.get("generated_at")
        or report.get("stored_at")
        or report.get("_blob_last_modified")
    )

    generated_iso = None
    generated_display = plan_data.get("generated_display")
    generated_unix = None
    if generated_dt:
        generated_iso = generated_dt.isoformat().replace("+00:00", "Z")
        generated_unix = int(generated_dt.timestamp())
        if not generated_display:
            generated_display = generated_dt.strftime("%b %d, %Y %H:%M UTC")

    strategies = plan_data.get("strategies")
    prepared_strategies: Dict[str, Any] = {}
    if isinstance(strategies, dict):
        labels = {
            "day_trading": "Day Trading",
            "swing_trading": "Swing Trading",
            "longterm_trading": "Long-Term Trading",
        }
        for key, label in labels.items():
            prepared = _prepare_strategy_for_dashboard(strategies.get(key))
            if prepared:
                prepared["label"] = label
                prepared_strategies[key] = prepared

    consensus = _extract_consensus_snapshot(report, symbol_clean)
    price_info, price_action = _extract_price_details(report)

    summary: Dict[str, Any] = {
        "symbol": symbol_clean,
        "generated_iso": generated_iso,
        "generated_display": generated_display,
        "generated_unix": generated_unix,
        "strategies": prepared_strategies,
        "consensus": consensus,
        "price": price_info,
        "price_action": price_action,
        "stored_at": report.get("stored_at"),
        "source": {
            "blob": report.get("_blob_name"),
            "user_id": report.get("user_id"),
            "ai_job_id": report.get("ai_job_id"),
        },
    }

    return summary

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