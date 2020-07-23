"""
This Script is build by Johannes Jacob
and published on johannesjacob.com

Heart Rate Zones Calculation:
Zone 1 low Less than 73% of LTHR | 20 TSS/hr
Zone 1 73% to 77% of LTHR | 30 TSS/hr
Zone 1 high 77% to 81% of LTHR | 40 TSS/hr
Zone 2 low 81% to 85% of LTHR | 50 TSS/hr
Zone 2 high 85% to 89% of LTHR | 60 TSS/hr
Zone 3 89% to 93% of LTHR | 70 TSS/hr
Zone 4 93% to 100% of LTHR | 80 TSS/hr
Zone 5a 100% to 103% of LTHR | 100 TSS/hr
Zone 5b 103% to 106% of LTHR | 120 TSS/hr
Zone 5c More than 106% of LTHR | 140 TSS/hr

"""

import os
import datetime
from fitparse import FitFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

lthr = 171.0  # Lactat Threshold Heart Rate Value
my_ftp = 311  # Functional Threshold Power
start_date = datetime.date(2019, 1, 1)
end_date = datetime.date.today()
directory = 'fitfiles'


def load_workout(workout_file):
    """
    Load fitfile and transforms
    it into a pandas Dataframe.
    Nan Values are replaced.
    :param workout_file:
    :return dataframe:
    """
    # Load the fitfile
    fitfile = FitFile(workout_file)

    # This is a ugly hack
    # to avoid timing issues
    while True:
        try:
            fitfile.messages
            break
        except KeyError:
            continue

    # Get all data messages that are of type record
    workout = []
    for record in fitfile.get_messages('record'):
        r = {}
        # Go through all the data entries in this record
        for record_data in record:
            r[record_data.name] = record_data.value

        workout.append(r)
    workout_df = pd.DataFrame(workout)
    workout_df.fillna(method='ffill', inplace=True)
    workout_df.fillna(method='backfill', inplace=True)
    return workout_df


def get_date(workout_df):
    """
    Gets the workout date.
    :param workout_df:
    :return date:
    """
    workout_date = workout_df['timestamp'][0].date()
    return workout_date


def get_tss(workout_df):
    """
    Calculates the TSS based on Power.
    :param workout_df:
    :return tss:
    """
    # Normalized Power
    norm_power = np.sqrt(np.sqrt(np.mean(workout_df['power'].rolling(30).mean() ** 4)))
    # Intensity
    intensity = norm_power / my_ftp
    # Moving time in seconds
    moving_time = int((workout_df['timestamp'].values[-1] - workout_df['timestamp'].values[0]) / 1000000000)
    # Trainings Stress Score
    workout_tss = (moving_time * norm_power * intensity) / (my_ftp * 3600.0) * 100.0
    return workout_tss


def get_hr_tss(workout_df):
    """
    Calculates the TSS based on Heart Rate.
    :param workout_df:
    :return hr_tss:
    """
    hr_zones = (pd.Series([0, 0.73, 0.77, 0.81, 0.85, 0.89, 0.93, 0.99, 1.03, 1.06, 2]) * lthr).to_list()
    workout_df['hrZone'] = pd.cut(workout_df['heart_rate'], hr_zones, labels=['Z1 low', 'Z1', 'Z1 high', 'Z2 low',
                                                                              'Z2 high', 'Z3', 'Z4', 'Z5a', 'Z5b',
                                                                              'Z5c'])
    workout_df['hrTSS'] = pd.cut(workout_df['heart_rate'], hr_zones,
                                 labels=[20, 30, 40, 50, 60, 70, 80, 100, 120, 140])
    workout_df['hrTSS'] = workout_df['hrTSS'].astype(int)
    hr_tss = np.sum(workout_df['hrTSS'].values) / 3600
    return hr_tss

# Create a PMC dataframe
# based on date range
date_range = pd.date_range(start_date, end_date).date
df = pd.DataFrame(index=date_range)

df['date'] = df.index
df['TSS'] = 0

# Loop through
# fitfile directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.fit'):
        workout = load_workout((os.path.join(directory, filename)))
        date = get_date(workout)
        if date not in df.index:
            # File not in date range
            continue
        if 'power' in workout:
            df.loc[date, 'TSS'] += get_tss(workout)
        elif 'heart_rate' in workout:
            df.loc[date, 'TSS'] += get_hr_tss(workout)
        else:
            # File does not contain power/HR
            continue

# Plot PMC
fig, ax = plt.subplots()
df['CTL'] = df['TSS'].rolling(42, min_periods=1).mean()
df['ATL'] = df['TSS'].rolling(7, min_periods=1).mean()
df['TSB'] = df['CTL'] - df['ATL']
# Plot CTL ATL and TSB
df[['CTL', 'ATL', 'TSB']].plot(ax=ax, title='Performance Management Chart')
plt.ylabel('CTL  /  ATL  /  TSB')
# Second y axis
ax2 = ax.twinx()
df[['TSS']].plot(ax=ax2, style='o')
plt.ylabel('TSS')
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# Upper and lower limits
# for y axis
plt.ylim(5, max(df['TSS'] + 5))
plt.show()
