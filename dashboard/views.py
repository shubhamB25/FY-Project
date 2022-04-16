from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from subprocess import run,PIPE
import sys

global flag;
Flag=True
def index(request):
    return render(request, 'dashboard/index.html')

def nypdlocation(request):
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as mtp
    import os
    # os.system('pip install -U scikit-learn')
    from sklearn.cluster import KMeans 
    from io import BytesIO
    import base64
    df=pd.read_csv("df_after_preprocessing2.csv")
    df.drop('Unnamed: 0', axis=1, inplace=True)
    #dropping unnecessary columns
    df.drop(['NYPD_PCT_CD','ADD_TS','DISP_TS','CLOSNG_TS','CIP_JOBS_Critical','CIP_JOBS_Non CIP','CIP_JOBS_Non Critical','CIP_JOBS_Serious'],axis=1,inplace=True)
    x=df[['Longitude','Latitude']]
    kmeans = KMeans(n_clusters=10, init='k-means++', random_state= 42)  
    y_predict= kmeans.fit_predict(x)
    print("\n\n\ny_predict",y_predict)
    x = np.array(x)
    mtp.figure(figsize=(10,7))
    mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 1, c = 'blue', label = 'Cluster 1') #for first cluster  
    mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 1, c = 'green', label = 'Cluster 2') #for second cluster  
    mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 1, c = 'red', label = 'Cluster 3') #for third cluster  
    mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 1, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
    mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 1, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
    mtp.scatter(x[y_predict == 5, 0], x[y_predict == 5, 1], s = 1, c = 'purple', label = 'Cluster 6') #for sixth cluster  
    mtp.scatter(x[y_predict == 6, 0], x[y_predict == 6, 1], s = 1, c = 'pink', label = 'Cluster 7') #for seventh cluster  
    mtp.scatter(x[y_predict == 7, 0], x[y_predict == 7, 1], s = 1, c = 'black', label = 'Cluster 8') #for Eight cluster  
    mtp.scatter(x[y_predict == 8, 0], x[y_predict == 8, 1], s = 1, c = 'beige', label = 'Cluster 9') #for ninth cluster  
    mtp.scatter(x[y_predict == 9, 0], x[y_predict == 9, 1], s = 1, c = 'brown', label = 'Cluster 10') #for tenth cluster  
    # mtp.scatter(x[y_predict == 10, 0], x[y_predict == 10, 1], s = 2, c = 'gray', label = 'Cluster 11') #for eleventh cluster  
    mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroid')   
    mtp.title('Clusters of locations')  
    mtp.axis('off')
    mtp.xlabel('Latitude')  
    mtp.ylabel('Longitude')   
    mtp.legend()
    buffer = BytesIO()
    mtp.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return render(request, 'dashboard/nypdlocation.html',{'plot':graph})

def nypdtype(request):
    import os
    # os.system('pip install geopandas')
    import pandas as pd
    import geopandas as gpd
    from shapely import wkt
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import io
    import urllib, base64
    from io import BytesIO
    df1=pd.read_csv("df_after_preprocessing2.csv")
    df2=pd.read_csv("NYPD_Calls_for_Service__Year_to_Date.csv")
    global df
    df = pd.merge(df1, df2, on='CAD_EVNT_ID', how='left')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(['Latitude_y','Longitude_y','RADIO_CODE_y'], axis=1, inplace=True)
    df.rename(columns = {'Latitude_x':'Latitude','Longitude_x':'Longitude','RADIO_CODE_x':'RADIO_CODE'}, inplace = True)
    return render(request, 'dashboard/nypdtype.html')

def twitter(request):
    return render(request, 'dashboard/twitter.html')

def newsfeed(request):
    return render(request, 'dashboard/newsfeed.html')

def plot(request):
    radio_code = request.POST.get('crimetype','')
    print('\n\nradio: ',radio_code)
    #radio_code='50G2'
    image=mlmodel(radio_code)
    print("\n\nOVERRRRRR\n\n")
    return render(request, 'dashboard/nypdtype.html',{'plot':image})

def mlmodel(radio_code):
    import matplotlib.pyplot as plt
    import urllib, base64
    from io import BytesIO
    radio=radio_code
    df_radio = df[df['RADIO_CODE']==(radio)]
    x = df_radio[['CAD_EVNT_ID','Latitude','Longitude','BORO_NM']]
    plt.switch_backend('AGG')
    plt.figure(figsize=(5,5))
    plt.title('Hello')
    import numpy as np
    import geopandas as gpd
    from shapely import wkt
    nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
    borough_count = x.groupby('BORO_NM').agg('count').reset_index()
    nyc.rename(columns={'BoroName':'BORO_NM'}, inplace=True)
    for i in nyc.index:
        nyc.at[i, "BORO_NM"] = nyc.at[i, "BORO_NM"].upper()
    bc_geo = nyc.merge(borough_count, on='BORO_NM')
    fig,ax = plt.subplots(1,1, figsize=(7,5))
    bc_geo.plot(column='CAD_EVNT_ID', cmap='viridis_r', alpha=.5, ax=ax, legend=True)
    bc_geo.apply(lambda x: ax.annotate(text='%s \n %.f'%(x.BORO_NM,x.CAD_EVNT_ID ), color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
    if radio_code=="22Q2":
        borough = 'Larceny'
    elif radio_code=="29H1":
        borough = 'Harassment'
    elif radio_code=="39T1":
        borough = 'Trespassing'
    elif radio_code=="20R":
        borough = 'Roberry'
    elif radio_code=="53D":
        borough = 'Dispute'
    elif radio_code=="24Q2":
        borough = 'Assault'
    elif radio_code=="50G2":
        borough = 'Disorderly Conduct'
    elif radio_code=="10S2":
        borough = 'Shooting'
    elif radio_code=="33":
        borough = 'Explosion'
    elif radio_code=="29Q1":
        borough = 'Other'
    plt.title("Number of Crimes by NYC Borough for Crime Type "+ borough)
    plt.axis('off')
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return(graph)


