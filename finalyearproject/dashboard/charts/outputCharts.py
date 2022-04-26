import pandas as pd
from os.path import exists
# from yaml import load

overview_file = "datasets/overview_stats.csv"

def loadData():
    df1=pd.read_csv("datasets/df_after_preprocessing2.csv")
    df2=pd.read_csv("datasets/NYPD_Calls_for_Service__Year_to_Date.csv")
    df = pd.merge(df1, df2, on='CAD_EVNT_ID', how='left')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(['Latitude_y','Longitude_y','RADIO_CODE_y'], axis=1, inplace=True)
    df.rename(columns = {'Latitude_x':'Latitude','Longitude_x':'Longitude','RADIO_CODE_x':'RADIO_CODE'}, inplace = True)
    return df



def saveGroupByToCsv(borough_count):
    borough_count.to_csv(overview_file)
def loadGroupByCsv():
    return pd.read_csv(overview_file)
def getBoroughStats():
    if not exists(overview_file) :
        df = loadData()
        df2 = df[['CAD_EVNT_ID','Latitude','Longitude','BORO_NM']]
        borough_count = df2.groupby('BORO_NM').agg('count').reset_index()
        saveGroupByToCsv(borough_count)
    else:
        borough_count = loadGroupByCsv()
    # print(borough_count)
    return borough_count


# RSS FEED dataset handling

import pandas as pd
from datetime import datetime,timedelta


# global df

def load_feed_data():
    df_old = pd.read_csv('datasets/newsdata.csv')
    df = df_old.iloc[83:] # removing invalid timestamps
    df['time'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['time'].dt.date
    df2 = pd.read_csv('datasets/news_borough_summary.csv')
    return [df,df2]


def get_time_series_data():
    datasets = load_feed_data()
    df = datasets[0]
    # filter by last 10 days
    today = datetime.now().date()
    delta = timedelta(days=10)
    a = today - delta
    data = {}
    for i in range(10,-1,-1):
        delta = timedelta(days=i)
        old_date = today - delta
        count = df[df['date'] == old_date].shape[0]

        # data[str(old_date)] = count
        data[old_date.strftime('%d %B')] = count
    df2 = datasets[1][['borough','count']]
    
    # print(df2.to_dict())  
    return [data,df2.to_dict('list')]


def get_news_folium_data():
    df = pd.read_csv('datasets/newsfeed_coordinates.csv')
    df = df[['Latitude','Longitude']]
    df = df.to_dict('list')
    return df