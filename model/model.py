import os
os.system('pip install geopandas')
import pandas as pd
import geopandas as gpd
from shapely import wkt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys
df1=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/df_after_preprocessing2.csv")
df2=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/NYPD_Calls_for_Service__Year_to_Date.csv")
df= pd.merge(df1, df2, on='CAD_EVNT_ID', how='left')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop(['Latitude_y','Longitude_y','RADIO_CODE_y'], axis=1, inplace=True)
df.rename(columns = {'Latitude_x':'Latitude','Longitude_x':'Longitude','RADIO_CODE_x':'RADIO_CODE'}, inplace = True)
df['RADIO_CODE'].value_counts()
radio=sys.argv[0]
#radio='50G2'
#radio=input('Enter Radio Code: ')
df_radio = df[df['RADIO_CODE']==(radio)]
x = df_radio[['CAD_EVNT_ID','Latitude','Longitude','BORO_NM']]
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
df_wm = nyc.to_crs(epsg=4326)
fig,ax = plt.subplots(1,1, figsize=(10,10))
base = df_wm.plot(color='white', edgecolor='black', ax=ax)
sns.scatterplot(x='Longitude', y='Latitude', hue='BORO_NM',s=20, data=x,)
print("DONEEEE")
