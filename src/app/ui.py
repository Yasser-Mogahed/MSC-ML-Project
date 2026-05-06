"""
Streamlit dashboard for customer churn prediction.

Interactive UI for submitting predictions and viewing results.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path so `src` imports work with `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.serving import predict  # noqa: E402
from src.utils import validate_input  # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.STREAMLIT_PAGE_CONFIG["page_title"],
    page_icon=settings.STREAMLIT_PAGE_CONFIG["page_icon"],
    layout=settings.STREAMLIT_PAGE_CONFIG["layout"],
    initial_sidebar_state=settings.STREAMLIT_PAGE_CONFIG["initial_sidebar_state"],
)

# ─────────────────────────────────────────────────────
# Custom CSS for dark theme
# ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --primary-color: #1f77b4;
        --background-color: #0e1117;
        --secondary-background-color: #1c1f26;
        --text-color: #e8eef2;
    }
    
    .metric-high { color: #ff6b6b; font-weight: bold; }
    .metric-low { color: #51cf66; font-weight: bold; }
    
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────


def render_header():
    """Render page header with title and description."""
    st.title("📊 Customer Churn Predictor")
    st.markdown(
        """
        Predict customer churn risk using advanced machine learning.
        Enter customer details below to get an instant prediction.
        """
    )
    st.divider()


def render_sidebar():
    """Render sidebar with configuration and info."""
    with st.sidebar:
        st.header("ℹ️ Overview")

        # Model information
        st.header("ℹ️ Model Information")
        st.markdown(
            f"""
            **Algorithm:** XGBoost Classifier  
            **Features:** {len(settings.FEATURE_NAMES)} input features  
            **Numeric:** {len(settings.NUMERIC_FEATURES)}  
            **Categorical:** {len(settings.CATEGORICAL_FEATURES)}  
            **Status:** ✅ Ready
            """
        )

        st.divider()

        # Usage guide
        with st.expander("📖 How to Use", expanded=False):
            st.markdown(
                """
                1. **Fill Form:** Enter customer details in all sections
                2. **Submit:** Click the "🔮 Predict Churn Risk" button
                3. **View Result:** See instant prediction with recommendations
                
                ### Input Guidelines
                - **Age:** 0-120 years
                - **Tenure:** 0-100 months
                - **Usage Frequency:** 0-100 scale
                - **Support Calls:** 0-50 calls
                - **Total Spend:** Any positive amount
                - **Last Interaction:** 0-365 days ago
                - **Payment Delay:** 0-100 days
                """
            )


def render_input_form() -> dict:
    """
    Render customer input form.

    Returns:
        Dictionary with all 10 customer features.
    """
    form_data = {}

    # ─────────────────────────────────────────────────────
    # Section 1: Customer Demographics
    # ─────────────────────────────────────────────────────
    st.subheader("👤 Customer Demographics")
    col1, col2 = st.columns([1, 1])

    with col1:
        form_data["Age"] = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
        )

    with col2:
        form_data["Gender"] = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=1,
        )

    # ─────────────────────────────────────────────────────
    # Section 2: Service & Usage
    # ─────────────────────────────────────────────────────
    st.subheader("📱 Service & Usage")
    col1, col2 = st.columns([1, 1])

    with col1:
        form_data["Subscription Type"] = st.selectbox(
            "Subscription Type",
            options=["Basic", "Standard", "Premium"],
            index=1,
        )

    with col2:
        form_data["Contract Length"] = st.selectbox(
            "Contract Length",
            options=["Monthly", "Quarterly", "Annual"],
            index=2,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        form_data["Usage Frequency"] = st.slider(
            "Usage Frequency (0-100)",
            min_value=0,
            max_value=100,
            value=14,
            step=1,
        )

    with col2:
        form_data["Support Calls"] = st.number_input(
            "Support Calls",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
        )

    # ─────────────────────────────────────────────────────
    # Section 3: Account & Billing
    # ─────────────────────────────────────────────────────
    st.subheader("💳 Account & Billing")
    col1, col2 = st.columns([1, 1])

    with col1:
        form_data["Tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
        )

    with col2:
        form_data["Total Spend"] = st.number_input(
            "Total Spend ($)",
            min_value=0.0,
            value=932.0,
            step=10.0,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        form_data["Last Interaction"] = st.number_input(
            "Last Interaction (days ago)",
            min_value=0,
            max_value=365,
            value=17,
            step=1,
        )

    with col2:
        form_data["Payment Delay"] = st.number_input(
            "Payment Delay (days)",
            min_value=0,
            max_value=100,
            value=18,
            step=1,
        )

    return form_data


def validate_form(data: dict) -> bool:
    """
    Validate form data using Pydantic.

    Args:
        data: Dictionary of form inputs.

    Returns:
        True if valid, False otherwise (shows error message).
    """
    is_valid, error_msg = validate_input(data)

    if not is_valid:
        st.error(f"⚠️ **Validation Error:** {error_msg}")
        return False

    return True


def render_prediction(result: dict):
    """
    Render prediction result card.

    Args:
        result: Prediction result dictionary from API.
    """
    if "error" in result:
        st.error(f"🚨 **Prediction Error:** {result['error']}")
        return

    prediction = result["prediction"]
    probability = result["churn_probability"]
    # ─────────────────────────────────────────────────────
    # Churn risk card
    # ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        # High churn risk
        st.container(border=True).markdown(
            """
            <div style="text-align: center;">
                <h2 style="color: #ff6b6b;">🔴 High Churn Risk</h2>
                <p style="font-size: 20px; color: #ff8787;">
                    This customer is at <strong>high risk</strong> of churning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%",
                delta=f"{(probability - 0.5) * 100:.1f}% above baseline",
            )

        with col2:
            risk_level = "Critical" if probability > 0.8 else "High"
            st.metric("Risk Level", risk_level)

        with col3:
            st.metric("Recommendation", "⚠️ Proactive Outreach")

        # Progress bar
        st.progress(probability, text=f"Risk Score: {probability:.2%}")

        # Retention strategies based on contract
        st.subheader("💡 Recommended Retention Strategies")
        contract = result.get("contract_type", "Unknown")

        if "Month" in contract:
            st.markdown(
                """
                **Monthly Contract Detected:**
                - Offer a 1-year or 2-year contract discount (15-25% savings)
                - Highlight stability and commitment benefits
                - Priority: **HIGH**
                """
            )
        elif "Quarterly" in contract:
            st.markdown(
                """
                **Quarterly Contract Detected:**
                - Offer upgrade to annual plan with 10% discount
                - Suggest premium tier with loyalty rewards
                - Priority: **MEDIUM**
                """
            )
        else:  # Annual
            st.markdown(
                """
                **Annual Contract Detected:**
                - Already committed; focus on upsell/cross-sell
                - Loyalty rewards or exclusive benefits
                - Priority: **LOW**
                """
            )

    else:
        # Low churn risk
        st.container(border=True).markdown(
            """
            <div style="text-align: center;">
                <h2 style="color: #51cf66;">🟢 Low Churn Risk</h2>
                <p style="font-size: 20px; color: #69db7c;">
                    This customer is likely to <strong>stay</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%",
                delta=f"{(probability - 0.5) * 100:.1f}% below baseline",
            )

        with col2:
            st.metric("Risk Level", "Low")

        with col3:
            st.metric("Action", "✅ Maintain")

        # Progress bar
        st.progress(1 - probability, text=f"Loyalty Score: {(1 - probability):.2%}")

        # Positive messaging
        st.markdown(
            """
            **✨ Great News!** This customer has strong engagement and satisfaction signals.
            
            ### Recommended Actions
            - Continue excellent service delivery
            - Consider loyalty programs or referral incentives
            - Identify upsell/cross-sell opportunities
            """
        )


def main():
    """Main application flow."""
    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Collect form input
    form_data = render_input_form()

    # Prediction button
    st.divider()
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # Validate input
        if not validate_form(form_data):
            st.stop()

        # Show spinner during prediction
        with st.spinner("🔄 Analyzing customer..."):
            try:
                result = predict(form_data)
            except Exception as e:
                result = {"error": str(e)}

        # Render prediction result
        render_prediction(result)


if __name__ == "__main__":
    main()
