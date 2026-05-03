import pandas as pd
df = pd.read_csv("data/fires.csv")
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['lat_bin'] = df['latitude'].round(1)
df['lon_bin'] = df['longitude'].round(1)

# Aggregate fire detections
fires = df.groupby(['lat_bin', 'lon_bin', 'acq_date'], as_index=False).agg({
    'frp': 'mean'
})

all_dates = pd.date_range(fires['acq_date'].min(), fires['acq_date'].max())

# All grid cells
grid_cells = fires[['lat_bin','lon_bin']].drop_duplicates()
full_grid = grid_cells.merge(
    pd.DataFrame({'acq_date': all_dates}),
    how='cross'
)

data = full_grid.merge(
    fires,
    on=['lat_bin','lon_bin','acq_date'],
    how='left'
)

data['fire_today'] = data['frp'].notna().astype(int)
data['frp'] = data['frp'].fillna(0)

data = data.sort_values(['lat_bin','lon_bin','acq_date']).reset_index(drop=True)

data['fire_last_1d'] = data.groupby(['lat_bin','lon_bin'])['fire_today'].shift(1)
data['fire_last_3d'] = data.groupby(['lat_bin','lon_bin'])['fire_today']\
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())

data['fire_last_7d'] = data.groupby(['lat_bin','lon_bin'])['fire_today']\
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
data['target'] = data.groupby(['lat_bin','lon_bin'])['fire_today'].shift(-1)
data = data.dropna().reset_index(drop=True)
X = data[['lat_bin','lon_bin','frp','fire_last_1d','fire_last_3d','fire_last_7d']]
y = data['target']