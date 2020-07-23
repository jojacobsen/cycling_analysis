from fitparse import FitFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the fitfile
fitfile = FitFile('2019-03-11-185427-ELEMNT BOLT 4043-167-0.fit')

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

df = pd.DataFrame(workout)

# Describe the workout
print(df[['power', 'heart_rate', 'speed', 'cadence']].describe())

# Plot it!
df_dropped = df.drop(['position_lat', 'position_long', 'distance', 'timestamp'], axis=1)
means = df_dropped.mean()
errors = df_dropped.std()
fig, ax = plt.subplots()

means.plot.bar(yerr=errors, ax=ax)

# Customize the grid
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


fig, ax = plt.subplots()
df[['power', 'heart_rate', 'cadence']].plot.hist(bins=100, alpha=0.5, range=(0, 400), ax=ax)
ax.legend()
# Customize the grid
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

fig, ax = plt.subplots()
df[['power', 'heart_rate', 'cadence']].plot(ax=ax)
ax.legend()
plt.xlabel("Seconds")
# Customize the grid
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


ftp = 275

# Normalized Power
norm_power = np.sqrt(np.sqrt(np.mean(df['power'].rolling(30).mean() ** 4)))

# Intensity
intensity = norm_power / ftp

moving_time = len(df)

# Trainings Stress Score
tss = (moving_time * norm_power * intensity) / (ftp * 3600.0) * 100.0

print('\n'
      'NP: {} W \n'
      'TSS: {} \n'
      'IF: {}'.format(str(round(norm_power, 1)), str(round(tss, 1)), str(round(intensity, 2))))
