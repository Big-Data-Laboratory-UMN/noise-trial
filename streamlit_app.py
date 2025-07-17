import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("Visualization - Monitoring")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df["Time"], format='%m/%d/%Y %H:%M:%S')
    df['seconds_elapsed'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()

    # Summary statistics (global)
    summary = df.groupby('Weight').agg({
        'MeaValue': ['mean', 'median', 'min', 'max', 'std', 'var'],
        'MeaValue.1': ['mean', 'median', 'min', 'max', 'std', 'var']
    }).round(2)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index(inplace=True)

    weights = df['Weight'].unique()

    # Loop over each weight
    for weight in weights:
        st.subheader(f"Noise Analysis - Weight {weight}")

        df_w = df[df['Weight'] == weight].copy()
        df_w['RecNoAdj'] = range(len(df_w))
        df_w = df_w[df_w['RecNoAdj'] < 600]

        # First 300 and last 300
        df_first = df_w[df_w['RecNoAdj'] < 300].copy()
        df_first['RecLocal'] = df_first['RecNoAdj']
        df_first['Segment'] = 'First 300'

        df_last = df_w[(df_w['RecNoAdj'] >= 300) & (df_w['RecNoAdj'] < 600)].copy()
        df_last['RecLocal'] = df_last['RecNoAdj'] - 300
        df_last['Segment'] = 'Last 300'

        df_combined = pd.concat([df_first, df_last])

        # Forecast using EMA
        df_forecast = df_w.set_index('datetime')
        df_numeric = df_forecast.select_dtypes(include='number')
        df_resampled = df_numeric.resample('s').mean().interpolate()
        ts = df_resampled['MeaValue.1']
        ema = ts.ewm(span=30, adjust=False).mean()
        forecast_value = ema.iloc[-1]
        forecast_index = pd.date_range(start=ts.index[-1] + timedelta(seconds=1), periods=300, freq='s')
        forecast = pd.Series([forecast_value] * 300, index=forecast_index)

        forecast_df = pd.DataFrame({
            'datetime': forecast.index,
            'MeaValue.1': forecast.values,
            'Weight': weight,
            'seconds_elapsed': (forecast.index - df['datetime'].min()).total_seconds()
        })

        # Plotting for this weight - now with 3 separate charts
        fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [2, 2, 2]})

        # Chart 1: First 300 records
        axes[0].plot(df_first['RecLocal'], df_first['MeaValue.1'], color='steelblue')
        axes[0].set_title(f'Noise - First 300 Records (Weight {weight})')
        axes[0].set_xlim([0, 300])
        axes[0].set_ylabel('Mea Value')
        axes[0].axhline(85, color='red', linestyle='--', linewidth=1.5, label='Threshold = 85')
        axes[0].legend()

        # Chart 2: Last 300 records
        axes[1].plot(df_last['RecLocal'], df_last['MeaValue.1'], color='darkorange')
        axes[1].set_title(f'Noise - Last 300 Records (Weight {weight})')
        axes[1].set_xlim([0, 300])
        axes[1].set_ylabel('Mea Value')
        axes[1].axhline(85, color='red', linestyle='--', linewidth=1.5, label='Threshold = 85')
        axes[0].legend()

        # Chart 3: Time Series with EMA Forecast
        axes[2].plot(ts.index, ts.values, color='steelblue', label='Observed')
        axes[2].plot(forecast_df['datetime'], forecast_df['MeaValue.1'], color='red', linestyle='--', label='EMA Forecast')
        axes[2].set_title(f'Noise Over Time (Seconds Elapsed) - Weight {weight}')
        axes[2].set_xlabel('Datetime')
        axes[2].set_ylabel('Mea Value')
        axes[2].tick_params(axis='x', rotation=25)
        axes[2].axhline(85, color='red', linestyle='--', linewidth=1.5, label='Threshold = 85')
        axes[2].legend()

        # Show chart
        fig.suptitle("Visualization - Monitoring", fontsize=30, fontweight='bold', y=1)
        st.pyplot(fig)

    # Summary for all weights at bottom
    st.subheader("Summary Statistics for All Weights")
    st.dataframe(summary)