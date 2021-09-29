import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
import numpy as np
import matplotlib
from pandas_profiling import ProfileReport
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter
import seaborn as sns


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('temporal.csv')
#df.head(10) #View first 10 data rows
df.head(10)
df.describe()
df.info()
print("done")

format_dict = {'data science':'${0:,.2f}', 'Mes':'{:%m-%Y}', 'machine learning':'{:.2%}'}
#We make sure that the Month column has datetime format
df['Mes'] = pd.to_datetime(df['Mes'])
#We apply the style to the visualization
df.head().style.format(format_dict)

df.head(10).style.format(format_dict).background_gradient(subset=['data science', 'machine learning'], cmap='BuGn')

df.head().style.format(format_dict).bar(color='red', subset=['data science', 'deep learning'])

#prof = ProfileReport(df)
#prof.to_file(output_file='report.html')

f = open("read.html",'w')
f.write(df.head().style.format(format_dict).bar(color='red', subset=['data science', 'deep learning']).render())
f.close()

#_______________________________________________________________
                    #How to use Seaborn
#_______________________________________________________________

sns.set()
#sns.scatterplot(df['Mes'], df['data science'])
#sns.relplot(x='Mes', y='deep learning', hue='data science', size='machine learning', col='categorical', data=df)
#sns.heatmap(df.corr(), annot=True, fmt='.2f')
#sns.pairplot(df)
#sns.pairplot(df, hue='categorical')
#sns.jointplot(x='data science', y='machine learning', data=df)
#sns.catplot(x='categorical', y='data science', kind='violin', data=df)

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
sns.scatterplot(x="Mes", y="deep learning", hue="categorical", data=df, ax=axes[0])
axes[0].set_title('Deep Learning')
sns.scatterplot(x="Mes", y="machine learning", hue="categorical", data=df, ax=axes[1])
axes[1].set_title('Machine Learning')

plt.show()

#_______________________________________________________________
                #How to use Matplotlib
#_______________________________________________________________


##plt.plot(df['Mes'], df['data science'], label='data science')
##plt.plot(df['Mes'], df['machine learning'], label='machine learning')
##plt.plot(df['Mes'], df['deep learning'], label='deep learning')
##plt.xlabel('Date')
##plt.ylabel('Popularity')
##plt.title('Popularity of AI terms by date')
##plt.grid(True)
##plt.legend()
##
##fig, axes = plt.subplots(2,2)
##axes[0, 0].hist(df['data science'])
##axes[0, 1].scatter(df['Mes'], df['data science'])
##axes[1, 0].plot(df['Mes'], df['machine learning'])
##axes[1, 1].plot(df['Mes'], df['deep learning'])
##
##plt.plot(df['Mes'], df['data science'], 'r-')
##plt.plot(df['Mes'], df['data science']*2, 'bs')
##plt.plot(df['Mes'], df['data science']*3, 'g^')
##
##plt.show()
##
##plt.scatter(df['data science'], df['machine learning'])
##
##plt.show()
##
##plt.bar(df['Mes'], df['machine learning'], width=20)
##
##plt.show()
##
##print("here")
##plt.plot(df['Mes'], df['data science'], label='data science')
##plt.plot(df['Mes'], df['machine learning'], label='machine learning')
##plt.plot(df['Mes'], df['deep learning'], label='deep learning')
##plt.xlabel('Date')
##plt.ylabel('Popularity')
##plt.title('Popularity of AI terms by date')
##plt.grid(True)
##plt.text(x='2010-01-01', y=80, s=r'$\lambda=1, r^2=0.8$') #Coordinates use the same units as the graph
##plt.annotate('Notice something?', xy=('2014-01-01', 30), xytext=('2006-01-01', 50), arrowprops={'facecolor':'red', 'shrink':0.05})
##
##plt.show()

