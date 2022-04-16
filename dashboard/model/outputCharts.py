import pandas as pd
from os.path import exists
# from yaml import load

overview_file = "overview_stats.csv"

def loadData():
    df1=pd.read_csv("df_after_preprocessing2.csv")
    df2=pd.read_csv("NYPD_Calls_for_Service__Year_to_Date.csv")
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


# getBoroughStats()

