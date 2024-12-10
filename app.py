import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import itertools
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import plotly.graph_objects as go
import warnings
import datetime

warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Cointegrated ETFs Strategy")

# Password login
password = st.text_input("Enter Password:", type="password")
if password != "d3r1ck!":
    st.warning("Please enter the correct password")
    st.stop()

st.success("Access Granted!")

st.markdown("""
<style>
    body {
        background: #f9f9f9;
        font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .stApp {
        background: #f9f9f9;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    header, footer {
        visibility: hidden;
    }
    .title {
        font-size: 2.5em;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1.5em;
    }
    .section-header {
        font-size: 1.5em;
        color: #222;
        border-bottom: 2px solid #eaeaea;
        padding-bottom: 0.2em;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Cointegrated ETFs Trading Strategy</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Explore and visualize cointegrated ETF pairs and backtest a pairs trading strategy.</div>", unsafe_allow_html=True)

# Today's date
today = datetime.date.today()
# For Parts A, B, D: 1 year + 1 day before today
one_year_one_day_ago = today - datetime.timedelta(days=366)
# For Part C: 6 months + 1 day before today ~ 184 days
six_months_one_day_ago = today - datetime.timedelta(days=184)

#--------------------------------------------
# Helper functions
#--------------------------------------------
def perform_adf_test(spread, significance_level=0.05):
    try:
        result = adfuller(spread.dropna())
        p_value = result[1]
    except:
        p_value = np.nan
    return p_value

def calculate_emrt(spread, C=2.0):
    theta = spread.mean()
    s = spread.std()
    is_max = (spread.shift(1)<spread)&(spread.shift(-1)<spread)
    is_min = (spread.shift(1)>spread)&(spread.shift(-1)>spread)
    important_max = is_max & ((spread - theta) >= C*s)
    important_min = is_min & ((theta - spread) >= C*s)
    extremes = spread[important_max | important_min].sort_index()
    tau = list(extremes.index)
    if len(tau)<2:
        return np.nan
    intervals = [(tau[i]-tau[i-1]).days for i in range(1,len(tau))]
    return np.mean(intervals) if intervals else np.nan

def construct_spread(price_df, s1, s2, B):
    if s1 not in price_df.columns or s2 not in price_df.columns:
        return pd.Series(dtype=float)
    return price_df[s1] - B*price_df[s2]

def backtest_strategy(price_data, s1, s2, B, transaction_cost=0.001, C=2.0, window=60):
    spread = construct_spread(price_data, s1, s2, B)
    df = spread.to_frame('Spread')
    df['Theta'] = df['Spread'].rolling(window).mean()
    df['Sigma'] = df['Spread'].rolling(window).std()
    df['Is_Max'] = (df['Spread'].shift(1)<df['Spread'])&(df['Spread'].shift(-1)<df['Spread'])
    df['Is_Min'] = (df['Spread'].shift(1)>df['Spread'])&(df['Spread'].shift(-1)>df['Spread'])
    df['Important_Max'] = df['Is_Max'] & ((df['Spread']-df['Theta'])>=C*df['Sigma'])
    df['Important_Min'] = df['Is_Min'] & ((df['Theta']-df['Spread'])>=C*df['Sigma'])

    trades = []
    open_trades = []
    for current_date, row in df.iterrows():
        if pd.isna(row['Theta']) or pd.isna(row['Sigma']):
            continue
        if row['Important_Max']:
            entry_price_a = price_data.loc[current_date,s1]
            entry_price_b = price_data.loc[current_date,s2]
            q_a = -1.0
            q_b = B
            trades.append({
                'Pair': f"{s1}-{s2}",
                'Entry_Date': current_date,
                'Action': 'Sell_A_Buy_B',
                'Entry_Price_A': entry_price_a,
                'Entry_Price_B': entry_price_b,
                'q_a': q_a,
                'q_b': q_b,
                'Exit_Date': None,
                'Exit_Price_A': None,
                'Exit_Price_B': None,
                'PnL': None
            })
            open_trades.append(len(trades)-1)
        elif row['Important_Min']:
            entry_price_a = price_data.loc[current_date,s1]
            entry_price_b = price_data.loc[current_date,s2]
            q_a = 1.0
            q_b = -B
            trades.append({
                'Pair': f"{s1}-{s2}",
                'Entry_Date': current_date,
                'Action': 'Buy_A_Sell_B',
                'Entry_Price_A': entry_price_a,
                'Entry_Price_B': entry_price_b,
                'q_a': q_a,
                'q_b': q_b,
                'Exit_Date': None,
                'Exit_Price_A': None,
                'Exit_Price_B': None,
                'PnL': None
            })
            open_trades.append(len(trades)-1)

        trades_to_close = []
        for idx in open_trades:
            t = trades[idx]
            q_a = t['q_a']
            q_b = t['q_b']
            entry_price_a = t['Entry_Price_A']
            entry_price_b = t['Entry_Price_B']

            if t['Action'] == 'Sell_A_Buy_B' and row['Spread']<=row['Theta']:
                exit_price_a = price_data.loc[current_date,s1]
                exit_price_b = price_data.loc[current_date,s2]
                pnl_a = q_a*(exit_price_a-entry_price_a)
                pnl_b = q_b*(exit_price_b-entry_price_b)
                pnl = pnl_a+pnl_b
                tc_a = transaction_cost*(abs(q_a)*entry_price_a+abs(q_a)*exit_price_a)
                tc_b = transaction_cost*(abs(q_b)*entry_price_b+abs(q_b)*exit_price_b)
                pnl-= (tc_a+tc_b)
                trades[idx]['Exit_Date'] = exit_date = current_date
                trades[idx]['Exit_Price_A'] = exit_price_a
                trades[idx]['Exit_Price_B'] = exit_price_b
                trades[idx]['PnL'] = pnl
                trades_to_close.append(idx)

            elif t['Action'] == 'Buy_A_Sell_B' and row['Spread']>=row['Theta']:
                exit_price_a = price_data.loc[current_date,s1]
                exit_price_b = price_data.loc[current_date,s2]
                pnl_a = q_a*(exit_price_a-entry_price_a)
                pnl_b = q_b*(exit_price_b-entry_price_b)
                pnl = pnl_a+pnl_b
                tc_a = transaction_cost*(abs(q_a)*entry_price_a+abs(q_a)*exit_price_a)
                tc_b = transaction_cost*(abs(q_b)*entry_price_b+abs(q_b)*exit_price_b)
                pnl -= (tc_a+tc_b)
                trades[idx]['Exit_Date'] = current_date
                trades[idx]['Exit_Price_A'] = exit_price_a
                trades[idx]['Exit_Price_B'] = exit_price_b
                trades[idx]['PnL'] = pnl
                trades_to_close.append(idx)

        open_trades = [o for o in open_trades if o not in trades_to_close]

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['PnL'].sum() if not trades_df.empty else 0.0
    return trades_df, total_pnl

def plot_spread(spread, pair, B):
    if spread.empty:
        return None
    theta = spread.mean()
    skewness = spread.skew()
    q_levels = {
        '2σ': (0.025,0.975),
        '2.5σ': (0.0062,0.9938),
        '3σ': (0.0013,0.9987)
    }
    quantiles = {}
    for label,(lq,uq) in q_levels.items():
        quantiles[label] = {
            'lower': spread.quantile(lq),
            'upper': spread.quantile(uq)
        }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread,
        mode='lines',
        name='Spread',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[theta]*len(spread),
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    for label,q in quantiles.items():
        style = 'dash' if label=='2σ' else 'dot' if label=='2.5σ' else 'dashdot'
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[q['upper']]*len(spread),
            mode='lines',
            name=f'Upper {label}',
            line=dict(color='green', dash=style)
        ))
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[q['lower']]*len(spread),
            mode='lines',
            name=f'Lower {label}',
            line=dict(color='orange', dash=style)
        ))
    fig.update_layout(
        title=f'Spread for {pair} | B={B:.4f}, Skew={skewness:.2f}',
        xaxis_title='Date',
        yaxis_title='Spread',
        hovermode='x unified',
        width=1000,
        height=600
    )
    return fig

def plot_spread_with_trades(price_data, s1, s2, B, trades_log, C=2.0, window=60):
    spread = construct_spread(price_data, s1, s2, B)
    df = spread.to_frame('Spread')
    df['Theta'] = df['Spread'].rolling(window=window).mean()
    df['Sigma'] = df['Spread'].rolling(window=window).std()
    df['Is_Max'] = (df['Spread'].shift(1)<df['Spread'])&(df['Spread'].shift(-1)<df['Spread'])
    df['Is_Min'] = (df['Spread'].shift(1)>df['Spread'])&(df['Spread'].shift(-1)>df['Spread'])
    df['Important_Max'] = df['Is_Max'] & ((df['Spread']-df['Theta'])>=C*df['Sigma'])
    df['Important_Min'] = df['Is_Min'] & ((df['Theta']-df['Spread'])>=C*df['Sigma'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread'], mode='lines', name='Spread', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Theta'], mode='lines', name='Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index[df['Important_Max']], y=df['Spread'][df['Important_Max']],
                             mode='markers', name='Important Max', marker=dict(color='orange',symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=df.index[df['Important_Min']], y=df['Spread'][df['Important_Min']],
                             mode='markers', name='Important Min', marker=dict(color='green', symbol='triangle-down')))

    for _, t in trades_log.iterrows():
        if pd.notna(t['Exit_Date']) and t['Entry_Date'] in df.index and t['Exit_Date'] in df.index:
            fig.add_trace(go.Scatter(
                x=[t['Entry_Date'], t['Exit_Date']],
                y=[df.loc[t['Entry_Date'],'Spread'], df.loc[t['Exit_Date'],'Spread']],
                mode='lines+markers',
                line=dict(color='purple', dash='dot'),
                marker=dict(color='purple', size=8),
                showlegend=False
            ))

    fig.update_layout(
        title=f"Spread & Trades for {s1}-{s2}",
        xaxis_title='Date',
        yaxis_title='Spread',
        hovermode='x unified',
        width=1000,
        height=600
    )
    return fig

############################################
# PART A LOGIC
############################################
st.markdown("<div class='section-header'>Part A: Identify ETF Pairs via SSD</div>", unsafe_allow_html=True)

partA_col1, partA_col2 = st.columns(2)
with partA_col1:
    start_date_A = st.date_input("Part A Start Date", value=one_year_one_day_ago)
with partA_col2:
    end_date_A = st.date_input("Part A End Date", value=today)

tickers = [
    'SPY', 'IVV', 'VOO', 'SPUU', 'SSO', 'UPRO', 'SPXL',
    'QQQ', 'QLD', 'TQQQ',
    'DIA', 'DDM', 'UDOW',
    'IWM', 'UWM', 'URTY', 'TNA',
    'MDY', 'MVV', 'MZZ',
    'IJR', 'SAA', 'SDD',
    'XLK', 'ROM', 'TECL',
    'XLF', 'UYG', 'FAS',
    'XLE', 'ERX', 'ERY',
    'XLV', 'CURE',
    'XLY', 'WANT',
    'XLP',
    'XLU', 'UTSL',
    'XLRE', 'DRN',
    'VNQ', 'REK',
    'XBI', 'LABU',
    'GDX', 'NUGT', 'DUST',
    'SMH', 'SOXL',
    'TLT', 'TMF',
    'USO', 'UCO', 'SCO',
    'UNG', 'UGA', 'BOIL', 'KOLD',
    'GLD', 'SLV', 'UGL', 'AGQ',
    'UUP', 'UDN', 'FXE', 'ULE', 'EUO', 'FXY', 'YCL', 'YCS', 'FXB', 'FXA', 'FXC',
    'VXX', 'UVXY', 'SVXY',
    'EEM', 'EET', 'EDC', 'EDZ',
    'EFA', 'EFU',
    'EWJ', 'EZJ', 'BBJP',
    'EWZ', 'UBR',
    'ASHR', 'FXI', 'XPP', 'YINN', 'CHAU',
    'ILF',
    'SH', 'SDS', 'QID', 'RWM',
    'FAZ', 'SAA', 'SDD',
]

if st.button("Run Part A"):
    st.write("Fetching price data for Part A...")
    price_data_A = yf.download(tickers, start=str(start_date_A), end=str(end_date_A))['Adj Close']
    price_data_A.dropna(inplace=True)
    st.write("Price data fetched successfully.")

    normalized_price = price_data_A.apply(zscore)
    stock_pairs = list(itertools.combinations(normalized_price.columns, 2))
    ssd_results = []
    for pair in stock_pairs:
        stock1, stock2 = pair
        spread = normalized_price[stock1] - normalized_price[stock2]
        ssd = np.sum(spread ** 2)
        ssd_results.append({
            'Pair': f'{stock1}-{stock2}',
            'SSD': ssd
        })

    ssd_df = pd.DataFrame(ssd_results)
    ssd_min = 1.0
    ssd_max = 5.0
    filtered_ssd_df = ssd_df[(ssd_df['SSD'] >= ssd_min) & (ssd_df['SSD'] <= ssd_max)]
    ssd_df_sorted = filtered_ssd_df.sort_values(by='SSD').reset_index(drop=True)
    top_n = 60
    selected_pairs = ssd_df_sorted.head(top_n)
    st.write(f"Number of Pairs with SSD between {ssd_min} and {ssd_max}: {len(filtered_ssd_df)}")
    st.write(f"Top {len(selected_pairs)} Pairs with Lowest SSD between {ssd_min} and {ssd_max}:")
    st.dataframe(selected_pairs)

    st.session_state['partA_selected_pairs'] = selected_pairs
    st.success("Part A completed successfully!")

############################################
# PART B LOGIC
############################################
st.markdown("<div class='section-header'>Part B: Optimize Cointegrated Pairs</div>", unsafe_allow_html=True)
CONFIG = {
    'C_THRESHOLD': 2.0,
    'VARIANCE_LIMIT': 40.0,
    'SIGNIFICANCE_LEVEL': 0.05,
    'B_RANGE_FINE': np.arange(-10.0, 10.01, 0.01),
    'BACKTEST_START_DATE': one_year_one_day_ago,
    'BACKTEST_END_DATE': today,
}

if st.button("Run Part B"):
    if 'partA_selected_pairs' not in st.session_state or st.session_state['partA_selected_pairs'].empty:
        st.error("Please run Part A first.")
    else:
        selected_pairs = st.session_state['partA_selected_pairs']
        unique_tickers_B = set()
        for pair in selected_pairs['Pair']:
            s1, s2 = pair.split('-')
            unique_tickers_B.update([s1,s2])
        unique_tickers_B = list(unique_tickers_B)

        st.write("Fetching price data for Part B...")
        price_data_B = yf.download(unique_tickers_B, start=str(CONFIG['BACKTEST_START_DATE']), end=str(CONFIG['BACKTEST_END_DATE']))['Adj Close']
        price_data_B.dropna(inplace=True)
        st.write("Price data fetched successfully.")

        results = []
        for _, row in selected_pairs.iterrows():
            pair = row['Pair']
            s1, s2 = pair.split('-')
            # Brute force over B
            best_B = None
            min_emrt = np.inf
            best_var = None
            for B in CONFIG['B_RANGE_FINE']:
                spread = construct_spread(price_data_B, s1, s2, B).dropna()
                if spread.empty:
                    continue
                emrt = calculate_emrt(spread, CONFIG['C_THRESHOLD'])
                var = spread.var()
                if np.isnan(emrt) or var>CONFIG['VARIANCE_LIMIT']:
                    continue
                p_value = perform_adf_test(spread, CONFIG['SIGNIFICANCE_LEVEL'])
                if np.isnan(p_value) or p_value>=CONFIG['SIGNIFICANCE_LEVEL']:
                    continue
                if emrt<min_emrt:
                    min_emrt = emrt
                    best_B = B
                    best_var = var

            if best_B is not None:
                # Store also Stock1/Stock2 for convenience
                results.append({
                    'Pair': pair,
                    'Stock1': s1,
                    'Stock2': s2,
                    'B': best_B,
                    'EMRT': min_emrt,
                    'Variance': best_var
                })

        optimized_df = pd.DataFrame(results)
        if not optimized_df.empty:
            optimized_df.sort_values(by='EMRT', inplace=True)
        st.write("Optimized Cointegrated Pairs:")
        st.dataframe(optimized_df)

        st.session_state['partB_optimized_pairs'] = optimized_df
        st.success("Part B completed successfully!")

############################################
# PART C LOGIC
############################################
st.markdown("<div class='section-header'>Part C: Visualize Selected Pairs</div>", unsafe_allow_html=True)
if 'partB_optimized_pairs' not in st.session_state or st.session_state['partB_optimized_pairs'].empty:
    st.info("Please run Part B first.")
else:
    optimized_pairs = st.session_state['partB_optimized_pairs']
    colC1, colC2 = st.columns(2)
    with colC1:
        start_date_C = st.date_input("Part C Start Date", value=six_months_one_day_ago)
    with colC2:
        end_date_C = st.date_input("Part C End Date", value=today)

    view_mode_C = st.radio("View Mode (Part C)", ["All Pairs", "Single Pair"])
    if view_mode_C == "Single Pair":
        selected_pair_C = st.selectbox("Select a pair to plot", optimized_pairs['Pair'].unique())
    else:
        selected_pair_C = None

    if st.button("Run Part C"):
        tickers_C = set()
        for pair in optimized_pairs['Pair']:
            s1, s2 = pair.split('-')
            tickers_C.update([s1,s2])
        tickers_C = list(tickers_C)
        st.write("Fetching data for Part C visualization...")
        price_data_C = yf.download(tickers_C, start=str(start_date_C), end=str(end_date_C))['Adj Close']
        price_data_C.dropna(inplace=True)
        st.write("Data fetched successfully.")

        if view_mode_C == "Single Pair" and selected_pair_C:
            row = optimized_pairs[optimized_pairs['Pair']==selected_pair_C].iloc[0]
            s1, s2, B = row['Stock1'], row['Stock2'], row['B']
            spread = construct_spread(price_data_C, s1, s2, B).dropna()
            fig = plot_spread(spread, selected_pair_C, B)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            for _, row in optimized_pairs.iterrows():
                pair = row['Pair']
                s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                spread = construct_spread(price_data_C, s1, s2, B).dropna()
                fig = plot_spread(spread, pair, B)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        st.success("Part C completed successfully!")

############################################
# PART D LOGIC
############################################
st.markdown("<div class='section-header'>Part D: Backtest the Strategy</div>", unsafe_allow_html=True)

if 'partB_optimized_pairs' not in st.session_state or st.session_state['partB_optimized_pairs'].empty:
    st.info("Please run Part B first.")
else:
    optimized_pairs_D = st.session_state['partB_optimized_pairs']
    colD1, colD2 = st.columns(2)
    with colD1:
        start_date_D = st.date_input("Part D Start Date", value=one_year_one_day_ago)
    with colD2:
        end_date_D = st.date_input("Part D End Date", value=today)

    TRANSACTION_COST = st.number_input("Transaction Cost (fraction)", 0.000, 0.01, 0.001)
    WINDOW_SIZE = st.number_input("Window Size (Days)", 1, 200, 60)
    view_mode_D = st.radio("View Mode (Part D)", ["All Pairs", "Single Pair"])
    if view_mode_D == "Single Pair":
        selected_pair_D = st.selectbox("Select a pair for backtest plot", optimized_pairs_D['Pair'].unique())
    else:
        selected_pair_D = None

    if st.button("Run Part D"):
        tickers_D = set()
        for pair in optimized_pairs_D['Pair']:
            s1,s2 = pair.split('-')
            tickers_D.update([s1,s2])
        tickers_D = list(tickers_D)
        st.write("Fetching price data for Part D...")
        price_data_D = yf.download(tickers_D, start=str(start_date_D), end=str(end_date_D))['Adj Close']
        price_data_D.dropna(inplace=True)
        st.write("Price data fetched successfully.")

        all_trades = []
        pair_pnl = {}
        for _, row in optimized_pairs_D.iterrows():
            pair = row['Pair']
            s1, s2, B = row['Stock1'], row['Stock2'], row['B']
            trades_log, total_profit = backtest_strategy(price_data_D, s1, s2, B,
                                                         transaction_cost=TRANSACTION_COST,
                                                         C=2.0,
                                                         window=WINDOW_SIZE)
            pair_pnl[pair] = total_profit
            if not trades_log.empty:
                all_trades.append(trades_log)

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()

        st.write("Total Profit/Loss for Each Pair:")
        for p, pnl in pair_pnl.items():
            st.write(f"{p}: ${pnl:.2f}")

        if view_mode_D == "Single Pair" and selected_pair_D:
            row = optimized_pairs_D[optimized_pairs_D['Pair']==selected_pair_D].iloc[0]
            s1, s2, B = row['Stock1'], row['Stock2'], row['B']
            if not combined_trades.empty:
                pair_trades = combined_trades[combined_trades['Pair']==selected_pair_D]
            else:
                pair_trades = pd.DataFrame()
            fig_bt = plot_spread_with_trades(price_data_D, s1, s2, B, pair_trades, C=2.0, window=WINDOW_SIZE)
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            if combined_trades.empty:
                st.write("No trades were executed based on the criteria. No plots to show.")
            else:
                for _, row in optimized_pairs_D.iterrows():
                    pair = row['Pair']
                    s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                    pair_trades = combined_trades[combined_trades['Pair']==pair]
                    fig_bt = plot_spread_with_trades(price_data_D, s1, s2, B, pair_trades, C=2.0, window=WINDOW_SIZE)
                    st.plotly_chart(fig_bt, use_container_width=True)

        st.success("Part D completed successfully!")

st.write("All parts (A, B, C, D) logic applied into Streamlit. End of app.")
