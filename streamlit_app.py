"""
Streamlit GUI for IRAM

This module provides a user-friendly web interface for interacting with the IRAM agent.
"""

import streamlit as st
import asyncio
import json

from src.agent_orchestrator import create_instagram_agent
from src.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="IRAM - Instagram Research Agent", layout="wide")

# Main app
def main():
    st.title("IRAM - Instagram Research Agent")

    # Initialize agent
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = create_instagram_agent()
            logger.info("IRAM agent initialized for Streamlit app")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return

    # Task input
    task = st.text_input("Enter your high-level task:", "Analyze the top 3 posts of @nasa")

    if st.button("Run Task"):
        if task:
            with st.spinner("Running task..."):
                try:
                    result = asyncio.run(st.session_state.agent.execute_task(task))
                    st.json(result)
                except Exception as e:
                    st.error(f"Task execution failed: {e}")
        else:
            st.warning("Please enter a task.")

if __name__ == "__main__":
    main()