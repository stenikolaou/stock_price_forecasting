import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load data
df = pd.read_csv('01_AMD.csv')
# Draw plot
plt.style.use('dark_background')
plot = df.plot(kind='area', x='Date', y='Close', color='white')
# Format plot title, x axis, Y axis
plt.title('Πορεία τιμής μετοχής AMD (1/1/2011 - 31/12/2020)', fontweight='bold', color='orange', fontsize ='20')
plt.xlabel('Ημερομηνία', labelpad=10, fontweight='bold', color='orange', fontsize='16')
plt.ylabel('Τιμή κλεισίματος σε $', labelpad=10, fontweight='bold', color='orange', fontsize='16')
# Draw horizontal axis lines
axes = plt.gca()
axes.get_legend().remove()
axes.yaxis.grid(color='grey', linestyle='')
# Format x axis ticks, y axis ticks
l = np.array(df['Date'])
plt.xticks(range(0, len(l), 120), l[::120], rotation=45, fontweight='bold', fontsize='8')
plot.yaxis.set_major_formatter(ticker.EngFormatter())
plt.yticks(fontweight='bold', fontsize='12')
# Show plot
plt.show()
