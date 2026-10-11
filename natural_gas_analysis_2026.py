"""EIA natural gas market analysis: East Region storage and Henry Hub spot prices.

Run in Spyder or terminal. Install: pip install pandas numpy matplotlib requests
Set EIA_API_KEY in environment or enter it at the secure prompt when asked.
Output is written to ./natural_gas_results/.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

BASE = 'https://api.eia.gov/v2/'
OUTPUT = Path('natural_gas_results')
START = '2024-01-01'
END = '2026-10-11'  # analysis cutoff; EIA may publish fewer observations
STORM_START = '2026-01-23'
STORM_END = '2026-01-27'


def fetch_eia(endpoint: str, api_key: str, frequency: str, facets: dict, start: str, end: str) -> pd.DataFrame:
    """Retrieve all pages, rather than silently truncating a time series."""
    records = []
    offset = 0
    while True:
        params = {
            'api_key': api_key, 'frequency': frequency, 'data[0]': 'value',
            'start': start, 'end': end, 'offset': offset, 'length': 5000,
            'sort[0][column]': 'period', 'sort[0][direction]': 'asc',
        }
        for field, value in facets.items():
            params[f'facets[{field}][]'] = value
        response = requests.get(BASE + endpoint, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json().get('response', {})
        batch = payload.get('data', [])
        records.extend(batch)
        total = int(payload.get('total', len(records)))
        if not batch or len(records) >= total:
            break
        offset += len(batch)
    if not records:
        raise ValueError(f'No EIA observations returned for {endpoint}. Check API key and facets.')
    data = pd.DataFrame(records)
    data['period'] = pd.to_datetime(data['period'])
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    return data.dropna(subset=['period', 'value']).sort_values('period').drop_duplicates('period')


def save_chart(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=250, bbox_inches='tight')
    plt.close()


def main():
    key = os.getenv('EIA_API_KEY') or getpass.getpass('Enter your EIA API key (hidden): ')
    if not key.strip():
        raise ValueError('An EIA API key is required.')
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # East Region total working gas (SWO), not the nonsalt-only series (SNO).
    storage = fetch_eia('natural-gas/stor/wkly/data/', key, 'weekly',
                        {'process': 'SWO', 'duoarea': 'R31'}, START, END)
    storage = storage[['period', 'value']].rename(columns={'value': 'storage_bcf'})
    storage['weekly_change'] = storage['storage_bcf'].diff()
    storage['season'] = storage['period'].dt.month.map(
        {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
         6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'})
    storage.to_csv(OUTPUT / 'east_region_storage.csv', index=False)

    plt.figure(figsize=(10, 4.5))
    plt.plot(storage['period'], storage['storage_bcf'], color='navy')
    plt.title('East Region working gas storage')
    plt.ylabel('Bcf'); plt.xlabel('Date'); plt.grid(alpha=.25)
    save_chart('east_region_storage.png')

    season = storage.groupby('season')['weekly_change'].agg(['mean','min','max','count'])
    season = season.reindex(['Winter','Spring','Summer','Fall'])
    season.to_csv(OUTPUT / 'seasonal_summary.csv')
    plt.figure(figsize=(8, 4.5))
    plt.bar(season.index, season['mean'])
    plt.axhline(0, color='black', linewidth=.8)
    plt.title('Mean weekly East Region storage change by season')
    plt.ylabel('Bcf/week'); plt.grid(axis='y', alpha=.2)
    save_chart('seasonal_storage_changes.png')

    prices = fetch_eia('natural-gas/pri/fut/data/', key, 'daily',
                       {'series': 'RNGWHHD'}, '2026-01-01', END)
    prices = prices[['period','value']].rename(columns={'value': 'henry_hub_price'})
    prices.to_csv(OUTPUT / 'henry_hub_daily_prices_2026.csv', index=False)
    peak = prices.loc[prices['henry_hub_price'].idxmax()]
    plt.figure(figsize=(10, 4.5))
    plt.plot(prices['period'], prices['henry_hub_price'], color='navy')
    plt.axvspan(pd.Timestamp(STORM_START), pd.Timestamp(STORM_END),
                color='orange', alpha=.2, label='Jan 23–27 analysis window')
    plt.scatter([peak['period']], [peak['henry_hub_price']], color='darkred', zorder=3)
    plt.annotate(f"Peak: ${peak['henry_hub_price']:.2f}/MMBtu",
                 (peak['period'], peak['henry_hub_price']), xytext=(12, -25),
                 textcoords='offset points', arrowprops={'arrowstyle':'->'})
    plt.title('Henry Hub spot prices — 2026'); plt.ylabel('$/MMBtu')
    plt.xlabel('Date'); plt.grid(alpha=.25); plt.legend()
    save_chart('henry_hub_2026.png')

    # Friday-ending weekly averages are aligned with Friday storage report dates.
    weekly = prices.set_index('period')['henry_hub_price'].resample('W-FRI').mean().reset_index()
    combined = storage.merge(weekly, on='period', how='inner')
    valid = combined.dropna(subset=['weekly_change','henry_hub_price']).copy()
    if len(valid) < 3 or valid['weekly_change'].nunique() < 2:
        raise ValueError('Not enough matching weekly observations for regression.')
    x = valid['weekly_change'].to_numpy()
    y = valid['henry_hub_price'].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    correlation = valid['weekly_change'].corr(valid['henry_hub_price'])
    r_squared = correlation ** 2
    combined['predicted_price'] = slope * combined['weekly_change'] + intercept
    combined['residual'] = combined['henry_hub_price'] - combined['predicted_price']
    combined.to_csv(OUTPUT / 'combined_weekly_analysis_2026.csv', index=False)

    excluded = valid.loc[~valid['period'].between('2026-01-23','2026-01-30')]
    without_corr = excluded['weekly_change'].corr(excluded['henry_hub_price'])
    summary = pd.DataFrame([{
        'matched_weeks': len(valid), 'correlation': correlation,
        'correlation_excluding_jan23_and_jan30': without_corr,
        'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
        'highest_daily_price': peak['henry_hub_price'],
        'highest_daily_price_date': peak['period'].date(),
        'note': 'Association only; this simple regression is not a causal or forecasting model.'
    }])
    summary.to_csv(OUTPUT / 'statistical_summary.csv', index=False)

    plt.figure(figsize=(9, 5.5))
    plt.scatter(x, y, alpha=.7, label='Weekly observations')
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, slope*xx+intercept, color='darkred', label=f'Linear fit (R²={r_squared:.3f})')
    plt.title('East Region storage changes vs Henry Hub weekly prices')
    plt.xlabel('Weekly storage change (Bcf)'); plt.ylabel('Henry Hub ($/MMBtu)')
    plt.grid(alpha=.2); plt.legend()
    save_chart('storage_price_regression.png')

    print(f'Finished. Files saved in: {OUTPUT.resolve()}')
    print(f'Matched weeks: {len(valid)} | r={correlation:.3f} | R²={r_squared:.3f}')
    print('Note: Price spike attribution requires independent weather and market evidence.')


if __name__ == '__main__':
    main()
