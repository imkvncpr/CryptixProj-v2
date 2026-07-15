import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")
st.title("Trading Dashboard")

def get_sample_data():
    return {
        'portfolio_value': [10000, 10500, 10200, 10800, 11200],
        'cash': 2000,
        'positions': [
            {'symbol': 'BTC', 'qty': 0.1, 'value': 4600},
            {'symbol': 'ETH', 'qty': 1.0, 'value': 3200}
        ],
        'trades': [
            {'date': '2024-01-01', 'symbol': 'BTC', 'pnl': 500},
            {'date': '2024-01-03', 'symbol': 'ETH', 'pnl': 100},
            {'date': '2024-01-05', 'symbol': 'BTC', 'pnl': -50}
        ]
    }

data = get_sample_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Metrics", "Trades", "Charts", "Risk"])

with tab1:
    st.header("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Value", "11200")
    with col2:
        st.metric("Cash Balance", "2000")
    with col3:
        st.metric("Open Positions", len(data['positions']))
    with col4:
        st.metric("Closed Trades", len(data['trades']))
    
    st.subheader("Open Positions")
    pos_df = pd.DataFrame(data['positions'])
    st.dataframe(pos_df, use_container_width=True)

with tab2:
    st.header("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", "1.45")
        st.metric("Sortino Ratio", "2.10")
    with col2:
        st.metric("Win Rate", "66.7%")
        st.metric("Max Drawdown", "5.2%")
    with col3:
        st.metric("Recovery Factor", "2.30")

with tab3:
    st.header("Trade History")
    trades_df = pd.DataFrame(data['trades'])
    st.dataframe(trades_df, use_container_width=True)

with tab4:
    st.header("Performance Charts")
    chart_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'Value': data['portfolio_value']
    })
    st.line_chart(chart_data.set_index('Date'))

with tab5:
    st.header("Risk Management")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Drawdown", "5.2%")
    with col2:
        st.metric("Daily Loss", "500")
    with col3:
        st.metric("Max Position Size", "500")

st.markdown("---")
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
