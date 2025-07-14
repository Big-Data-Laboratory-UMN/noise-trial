import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("Visualization - Monitoring")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read and prepare data
    df = pd.read_excel(uploaded_file, sheet_name="Noise Sample")

    st.write(df.head())  # Preview the first rows

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df["Time"], format='%m/%d/%Y %H:%M:%S')

    # Create a seconds_elapsed column for plotting over time
    df['seconds_elapsed'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()
    
     # Prepare combined chart data per weight
    # Separate by Weight
    df_A = df[df['Weight'] == 'A'].copy()
    df_B = df[df['Weight'] == 'B'].copy()
    df_A['RecNoAdj'] = range(len(df_A))
    df_B['RecNoAdj'] = range(len(df_B))

    df_A1 = df_A[df_A['RecNoAdj'] < 300].copy()
    df_A1['RecLocal'] = df_A1['RecNoAdj']
    df_A1['Segment'] = 'First 300'

    df_A2 = df_A[(df_A['RecNoAdj'] >= 300) & (df_A['RecNoAdj'] < 600)].copy()
    df_A2['RecLocal'] = df_A2['RecNoAdj'] - 300
    df_A2['Segment'] = 'Next 300'

    df_B1 = df_B[df_B['RecNoAdj'] < 300].copy()
    df_B1['RecLocal'] = df_B1['RecNoAdj']
    df_B1['Segment'] = 'First 300'

    df_B2 = df_B[(df_B['RecNoAdj'] >= 300) & (df_B['RecNoAdj'] < 600)].copy()
    df_B2['RecLocal'] = df_B2['RecNoAdj'] - 300
    df_B2['Segment'] = 'Next 300'

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

    # Fix for seconds_elapsed issue
    start_time = df['datetime'].min()
    forecast_df_A['seconds_elapsed'] = (forecast_df_A['datetime'] - start_time).dt.total_seconds()
    forecast_df_B['seconds_elapsed'] = (forecast_df_B['datetime'] - start_time).dt.total_seconds()

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(24, 26), gridspec_kw={'height_ratios': [2, 2, 2, 2, 1]})
    fig.suptitle("Visualization - Monitoring", fontsize=34, fontweight='bold', y=0.95)

    # Chart 1: Weight A - 300 first and next (split but same x-axis 0-299)
    axes[0].plot(df_A1['RecLocal'], df_A1['MeaValue.1'], label='First 300', color='purple')
    axes[0].plot(df_A2['RecLocal'], df_A2['MeaValue.1'], label='Next 300', color='blue')
    axes[0].set_title('Noise by Record No (0–299 Twice) - Weight A')
    axes[0].set_xlim([0, 300])
    axes[0].set_ylabel('Mea Value')
    axes[0].legend()

    # Chart 2: Weight B (split line)
    axes[1].plot(df_B1['RecLocal'], df_B1['MeaValue.1'], label='First 300', color='orange')
    axes[1].plot(df_B2['RecLocal'], df_B2['MeaValue.1'], label='Next 300', color='green')
    axes[1].set_title('Noise by Record No (0–299 Twice) - Weight B')
    axes[1].set_xlim([0, 300])
    axes[1].set_ylabel('Mea Value')
    axes[1].legend()

    # Chart 3: A by seconds_elapsed
    sns.lineplot(ax=axes[2], data=df_A, x='seconds_elapsed', y='MeaValue.1', color='steelblue', label='Observed A')
    sns.lineplot(ax=axes[2], data=forecast_df_A, x='seconds_elapsed', y='MeaValue.1', linestyle='--', color='red', label='Forecast A')
    axes[2].set_title('Noise Over Time (Seconds Elapsed) - Weight A')
    axes[2].set_xlabel('Seconds')
    axes[2].set_ylabel('Mea Value')
    axes[2].legend()

    # Chart 4: B by seconds_elapsed
    sns.lineplot(ax=axes[3], data=df_B, x='seconds_elapsed', y='MeaValue.1', color='darkorange', label='Observed B')
    sns.lineplot(ax=axes[3], data=forecast_df_B, x='seconds_elapsed', y='MeaValue.1', linestyle='--', color='blue', label='Forecast B')
    axes[3].set_title('Noise Over Time (Seconds Elapsed) - Weight B')
    axes[3].set_xlabel('Seconds')
    axes[3].set_ylabel('Mea Value')
    axes[3].legend()

    # Chart 5: Summary stats
    axes[4].axis('off')
    summary_text = summary.to_string(index=False)
    axes[4].text(0.01, 0.85, "Summary Statistics by Weight", fontsize=14, fontweight='bold')
    axes[4].text(0.01, 0.45, summary_text, family='monospace', fontsize=12)

    st.pyplot(fig)