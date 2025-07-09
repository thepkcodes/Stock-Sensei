"""
Utility functions for the Ultimate Stock Market AI Agent Streamlit App
"""

import streamlit as st
import time
import traceback
from functools import wraps
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadingState:
    """Context manager for loading states"""
    def __init__(self, message="Loading...", key=None):
        self.message = message
        self.key = key or message
        self.placeholder = None
        
    def __enter__(self):
        self.placeholder = st.empty()
        self.placeholder.info(f"⏳ {self.message}")
        if self.key:
            st.session_state.loading_state[self.key] = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.placeholder.empty()
        if self.key and self.key in st.session_state.loading_state:
            del st.session_state.loading_state[self.key]

def handle_error(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Add to session state errors
            if 'error_messages' not in st.session_state:
                st.session_state.error_messages = []
            
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'function': func.__name__,
                'error': str(e),
                'type': type(e).__name__
            }
            st.session_state.error_messages.append(error_details)
            
            # Display user-friendly error
            st.error(f"❌ {error_msg}")
            
            with st.expander("🔍 Error Details"):
                st.json(error_details)
                st.code(traceback.format_exc())
            
            return None
    return wrapper

def show_progress(message, steps):
    """Show progress bar for multi-step operations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update(step, total=None):
        if total is None:
            total = len(steps)
        progress = (step + 1) / total
        progress_bar.progress(progress)
        if step < len(steps):
            status_text.text(f"{message}: {steps[step]}")
        time.sleep(0.1)  # Small delay for visual effect
    
    return update

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def create_info_card(title, content, card_type="info"):
    """Create a styled information card"""
    colors = {
        "info": "#3498db",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "error": "#e74c3c",
        "primary": "#667eea"
    }
    
    color = colors.get(card_type, colors["info"])
    
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    ">
        <h4 style="margin: 0; color: {color};">{title}</h4>
        <p style="margin: 0.5rem 0 0 0; color: #333;">{content}</p>
    </div>
    """

def validate_ticker(ticker):
    """Validate ticker symbol"""
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty")
    
    ticker = ticker.upper().strip()
    
    # Basic validation
    if not ticker.isalpha():
        raise ValueError("Ticker symbol should only contain letters")
    
    if len(ticker) > 5:
        raise ValueError("Ticker symbol seems too long")
    
    return ticker

def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta > 0 else "red" if delta < 0 else "gray"
        if delta_color == "inverse":
            color = "red" if delta > 0 else "green" if delta < 0 else "gray"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_html = f'<p style="color: {color}; font-size: 0.9rem; margin: 0; font-weight: 600;">{arrow} {abs(delta):.2f}%</p>'
    
    return f"""
    <div class="metric-box" style="text-align: center;">
        <h3 style="margin: 0; color: #1f2937; font-weight: 800; font-size: 1.75rem; text-align: center;">{value}</h3>
        <p style="color: #6b7280; font-weight: 600; font-size: 0.9rem; margin: 0.5rem 0 0 0; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;">{label}</p>
        {delta_html}
    </div>
    """

def check_market_hours():
    """Check if market is open"""
    from datetime import datetime, time
    import pytz
    
    # Get current time in EST/EDT
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Market hours: 9:30 AM - 4:00 PM EST, Monday-Friday
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    is_market_hours = market_open <= now.time() <= market_close
    
    return {
        'is_open': is_weekday and is_market_hours,
        'current_time': now.strftime('%I:%M %p EST'),
        'day': now.strftime('%A'),
        'date': now.strftime('%Y-%m-%d')
    }

def create_disclaimer():
    """Create disclaimer text"""
    return """
    <div style="
        background: #f0f0f0;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #666;
    ">
        <strong>⚠️ Disclaimer:</strong> This AI-powered analysis is for educational and informational purposes only. 
        It should not be considered as financial advice. Always do your own research and consult with a 
        qualified financial advisor before making investment decisions.
    </div>
    """

def format_timeframe(days):
    """Format days into human-readable timeframe"""
    if days == 1:
        return "1 day"
    elif days < 7:
        return f"{days} days"
    elif days == 7:
        return "1 week"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    elif days == 30:
        return "1 month"
    else:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''}"

# Export all utilities
__all__ = [
    'LoadingState',
    'handle_error',
    'show_progress',
    'format_large_number',
    'create_info_card',
    'validate_ticker',
    'create_metric_card',
    'check_market_hours',
    'create_disclaimer',
    'format_timeframe'
]
