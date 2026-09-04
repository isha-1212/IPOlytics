"""Simple Streamlit UI for the IPOlytics FastAPI backend."""
import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
DOCUMENT_NAME = "Tempsens Instruments India Limited Prospectus"

st.set_page_config(page_title="IPOlytics", page_icon="📈", layout="centered")


def error_message(response: requests.Response) -> str:
    """Return a useful FastAPI error without showing a traceback."""
    try:
        detail = response.json().get("detail")
        if isinstance(detail, list):
            return "Please check the values entered and try again."
        if detail:
            return str(detail)
    except (ValueError, AttributeError):
        pass
    return "The server could not complete the request. Please try again."


def show_connection_error(error: requests.RequestException) -> None:
    if isinstance(error, requests.Timeout):
        st.error("The request timed out. Please try again in a moment.")
    else:
        st.error("Cannot reach FastAPI. Start the backend and try again.")


st.title("IPOlytics")
st.caption("AI-Powered IPO Listing Prediction & Research Assistant")

prediction_tab, research_tab = st.tabs(["IPO Prediction", "IPO Research"])

with prediction_tab:
    with st.form("prediction_form"):
        ipo_date = st.date_input("IPO Date")
        issue_size = st.number_input("Issue Size", min_value=0.01, value=500.0)
        qib = st.number_input("QIB Subscription", min_value=0.0, value=0.0)
        hni = st.number_input("HNI Subscription", min_value=0.0, value=0.0)
        rii = st.number_input("RII Subscription", min_value=0.0, value=0.0)
        total = st.number_input("Total Subscription", min_value=0.01, value=0.01)
        offer_price = st.number_input("Offer Price", min_value=0.01, value=1.0)
        predict = st.form_submit_button("Predict Listing", type="primary")

    if predict:
        payload = {
            "date": ipo_date.isoformat(),
            "Issue_Size": issue_size,
            "QIB": qib,
            "HNI": hni,
            "RII": rii,
            "Total": total,
            "Offer_Price": offer_price,
        }
        try:
            with st.spinner("Generating prediction..."):
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
            if response.ok:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
                positive, negative = st.columns(2)
                positive.metric("Positive Listing Probability", f"{result['positive_listing_probability']:.0%}")
                negative.metric("Negative Listing Probability", f"{result['negative_listing_probability']:.0%}")
            else:
                st.error(error_message(response))
        except requests.RequestException as error:
            show_connection_error(error)

    st.caption("Model-based estimate for educational/research purposes only. Not financial advice.")

with research_tab:
    st.write("IPO Document:")
    st.info(DOCUMENT_NAME)
    question = st.text_input(
        "Ask a question",
        placeholder="What are the major risks mentioned in this IPO?",
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Enter a question before asking.")
        else:
            try:
                with st.spinner("Searching the IPO document..."):
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"question": question.strip()},
                        timeout=90,
                    )
                if response.ok:
                    result = response.json()
                    st.subheader("Answer")
                    st.write(result["answer"])
                    st.subheader("Sources")
                    for source in result.get("sources", []):
                        st.write(f"- {source['source']} — Page {source['page']}")
                else:
                    st.error(error_message(response))
            except requests.RequestException as error:
                show_connection_error(error)
