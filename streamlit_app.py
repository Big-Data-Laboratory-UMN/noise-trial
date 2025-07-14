import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("Noise Forecast Dashboard (EMA)")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read and prepare data
    df = pd.read_excel(uploaded_file, sheet_name="Noise Sample")

    st.write(df.head())  # Preview the first rows

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df["Time"], format='%m/%d/%Y %H:%M:%S')



    # Add adjusted record number
    df_A = df[df['Weight'] == 'A'].copy()
    df_B = df[df['Weight'] == 'B'].copy()
    df_A['RecNoAdj'] = range(len(df_A))
    df_B['RecNoAdj'] = range(len(df_B))
    df_A = df_A[df_A['RecNoAdj'] < 600]
    df_B = df_B[df_B['RecNoAdj'] < 600]

    # Summary statistics
    summary = df.groupby('Weight').agg({
        'MeaValue': ['mean', 'median', 'min', 'max', 'std', 'var'],
        'MeaValue.1': ['mean', 'median', 'min', 'max', 'std', 'var']
    }).round(2)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index(inplace=True)

    # Forecast using Exponential Moving Average
    results = {}
    for weight in ['A', 'B']:
        df_w = df[df['Weight'] == weight].copy()
        df_w = df_w.set_index('datetime')
        df_w_numeric = df_w.select_dtypes(include='number')
        df_w_resampled = df_w_numeric.resample('s').mean().interpolate()
        ts = df_w_resampled['MeaValue.1']
        ema = ts.ewm(span=30, adjust=False).mean()
        forecast_value = ema.iloc[-1]
        forecast_index = pd.date_range(start=ts.index[-1] + timedelta(seconds=1), periods=300, freq='s')
        forecast = pd.Series([forecast_value] * 300, index=forecast_index)
        results[weight] = {'observed': ts, 'forecast': forecast}

    # Prepare forecast DataFrames
    forecast_df_A = pd.DataFrame({
        'datetime': results['A']['forecast'].index,
        'MeaValue.1': results['A']['forecast'].values,
        'Weight': 'A'
    })
    forecast_df_B = pd.DataFrame({
        'datetime': results['B']['forecast'].index,
        'MeaValue.1': results['B']['forecast'].values,
        'Weight': 'B'
    })

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(24, 22), gridspec_kw={'height_ratios': [2, 2, 2, 2, 1]})

    sns.lineplot(ax=axes[0], data=df_A, x='RecNoAdj', y='MeaValue.1', marker='o', color='steelblue')
    axes[0].set_title('Noise by Record No - Weight A')
    axes[0].set_xlim([0, 600])
    axes[0].set_ylabel('Mea Value')

    sns.lineplot(ax=axes[1], data=df_B, x='RecNoAdj', y='MeaValue.1', marker='o', color='darkorange')
    axes[1].set_title('Noise by Record No - Weight B')
    axes[1].set_xlim([0, 600])
    axes[1].set_ylabel('Mea Value')

    sns.lineplot(ax=axes[2], data=df[df['Weight'] == 'A'], x='datetime', y='MeaValue.1', marker='o', color='steelblue', label='Observed A')
    sns.lineplot(ax=axes[2], data=forecast_df_A, x='datetime', y='MeaValue.1', linestyle='--', color='red', label='Forecast A')
    axes[2].set_title('Noise by Time - Weight A')
    axes[2].set_ylabel('Mea Value')
    axes[2].tick_params(axis='x', rotation=25)
    axes[2].legend()

    sns.lineplot(ax=axes[3], data=df[df['Weight'] == 'B'], x='datetime', y='MeaValue.1', marker='o', color='darkorange', label='Observed B')
    sns.lineplot(ax=axes[3], data=forecast_df_B, x='datetime', y='MeaValue.1', linestyle='--', color='blue', label='Forecast B')
    axes[3].set_title('Noise by Time - Weight B')
    axes[3].set_ylabel('Mea Value')
    axes[3].tick_params(axis='x', rotation=25)
    axes[3].legend()

    axes[4].axis('off')
    summary_text = summary.to_string(index=False)
    axes[4].text(0.01, 0.85, "Summary Statistics by Weight", fontsize=14, fontweight='bold')
    axes[4].text(0.01, 0.45, summary_text, family='monospace', fontsize=12)

    # Render chart in Streamlit
    st.pyplot(fig)
