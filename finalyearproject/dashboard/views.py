from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from subprocess import run,PIPE
import sys
from collections import OrderedDict
import json


def index(request):
    return render(request, 'dashboard/index.html')

def nypdlocation(request):
    import os
    #os.system('pip install -U scikit-learn')
    import pandas as pd
    import matplotlib.pyplot as mtp
    #from sklearn.cluster import KMeans
    from io import BytesIO
    import base64
    import numpy as np
    from matplotlib.patches import Patch
    # os.system('pip install geopandas')
    # df=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/df_after_preprocessing2.csv")
    # df2=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/NYPD_Calls_for_Service__Year_to_Date.csv")
    # df= pd.merge(df, df2, on='CAD_EVNT_ID', how='left')
    # df.drop(['Latitude_y','Longitude_y','RADIO_CODE_y','Unnamed: 0'], axis=1, inplace=True)
    # df.rename(columns = {'Latitude_x':'Latitude','Longitude_x':'Longitude','RADIO_CODE_x':'RADIO_CODE'}, inplace = True)
    global x
    # x=df[['Longitude','Latitude']]
    # x.to_csv('crime_location_csv.csv')
    x=pd.read_csv('datasets/crime_location_csv.csv')
    x.drop('Unnamed: 0',inplace=True,axis=1)
    x =x.values 
    # kmeans = KMeans(n_clusters=10, init='k-means++', random_state= 42) 
    global y_predict
    # y_predict= kmeans.fit_predict(x) 
    # pd.DataFrame(y_predict).to_csv('y_predict.csv')
    y_predict=np.array(pd.read_csv('datasets/y_predict.csv')['0'])
    global kmeans_cluster_centers_0
    global kmeans_cluster_centers_1
    kmeans_cluster_centers_0=np.array(pd.read_csv('datasets/kmeans_cluster_centers_0.csv')['0'])
    kmeans_cluster_centers_1=np.array(pd.read_csv('datasets/kmeans_cluster_centers_1.csv')['0'])
    # print()
    # print()
    # print()
    # print()
    # print(y_predict)
    
    # print()
    # print()
    # print()
    # clusteroption="one"
    # dict_graphs=nypdloc_mlmodel(clusteroption)
    mtp.figure(figsize=(10,7))
    mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 2, c = 'blue', label = 'Cluster 1') #for first cluster  
    mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 2, c = 'green', label = 'Cluster 2') #for second cluster  
    mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 2, c = 'red', label = 'Cluster 3') #for third cluster  
    mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 2, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
    mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 2, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
    mtp.scatter(x[y_predict == 5, 0], x[y_predict == 5, 1], s = 2, c = 'purple', label = 'Cluster 6') #for sixth cluster  
    mtp.scatter(x[y_predict == 6, 0], x[y_predict == 6, 1], s = 2, c = 'pink', label = 'Cluster 7') #for seventh cluster  
    mtp.scatter(x[y_predict == 7, 0], x[y_predict == 7, 1], s = 2, c = 'orange', label = 'Cluster 8') #for Eight cluster  
    mtp.scatter(x[y_predict == 8, 0], x[y_predict == 8, 1], s = 2, c = 'yellowgreen', label = 'Cluster 9') #for ninth cluster  
    mtp.scatter(x[y_predict == 9, 0], x[y_predict == 9, 1], s = 2, c = 'brown', label = 'Cluster 10') #for tenth cluster   
    mtp.scatter(kmeans_cluster_centers_0, kmeans_cluster_centers_1, s = 100, c = 'yellow', label = 'Centroid')     
    mtp.xlabel('Longitude')  
    mtp.ylabel('Latitude')   
    legend_elements=[Patch(facecolor='blue',label='Region 1'),Patch(facecolor='green',label='Region 2'),
    Patch(facecolor='red',label='Region 3'),Patch(facecolor='cyan',label='Region 4'),
    Patch(facecolor='magenta',label='Region 5'),Patch(facecolor='purple',label='Region 6'),
    Patch(facecolor='pink',label='Region 7'),Patch(facecolor='orange',label='Region 8'),
    Patch(facecolor='yellowgreen',label='Region 9'),Patch(facecolor='brown',label='Region 10')]
    mtp.legend(handles=legend_elements,loc='upper left') 
    mtp.axis('off')
    buffer = BytesIO()
    mtp.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    global graph
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close() 
    #making merged dataset
    global merged
    merged = pd.read_csv("datasets/merged_with_categories.csv")
    global total_crimes
    total_crimes=merged.shape[0]
    #merged = pd.DataFrame(x,columns=['Latitude','Longitude'])
    global df_groupby_labels
    df_groupby_labels = merged.groupby('Labels')
    global chartdata
    global chartlabels
    chartdata=[len(df_groupby_labels.get_group(0)),len(df_groupby_labels.get_group(1)),len(df_groupby_labels.get_group(2)),len(df_groupby_labels.get_group(3)),len(df_groupby_labels.get_group(4)),len(df_groupby_labels.get_group(5)),len(df_groupby_labels.get_group(6)),len(df_groupby_labels.get_group(7)),len(df_groupby_labels.get_group(8)),len(df_groupby_labels.get_group(9))]
    chartlabels=['Region 1',"Region 2","Region 3","Region 4","Region 5","Region 6","Region 7","Region 8","Region 9","Region 10",]
    #Time between crime added to system and the time at which unit arrived
    merged['INCIDENT_DATE_TIME'] = pd.to_datetime(merged['INCIDENT_DATE_TIME'])
    merged['ARRIVD_TS'] = pd.to_datetime(merged['ARRIVD_TS'])
    merged["Time_Difference"] =  merged["ARRIVD_TS"] - merged["INCIDENT_DATE_TIME"]
    import datetime
    global avg_time_diff
    avg_time_diff = []
    for i in range(0,10):
        temp = df_groupby_labels.get_group(i)
        avg_time_diff.append((temp["Time_Difference"].mean().total_seconds())/60)
    return render(request, 'dashboard/nypdlocation.html',{'mainplot':graph,'chartdata':json.dumps(chartdata),'chartlabels':json.dumps(chartlabels),'avg_time_diff':json.dumps(avg_time_diff)})

def cluster(request):
    clusteroption = request.POST.get('clusteroption','')
    #clusteroption="one"
    result=nypd_clustermodel(clusteroption)
    cluster_graph=result[0]
    first_crime=result[1]
    second_crime=result[2]
    third_crime=result[3]
    location_address=result[4]
    first_crime_no=result[5]
    second_crime_no=result[6]
    third_crime_no=result[7]
    total_crimes_percent=result[8]
    region=result[9]
    first_crime_percent=result[10]
    second_crime_percent=result[11]
    third_crime_percent=result[12]
    total_crimes=result[13]
    vertical_dist=result[14]
    horizontal_dist=result[15]
    return render(request, 'dashboard/nypdlocation.html',{'mainplot':graph,'clusterplot':cluster_graph,
    'chartdata':json.dumps(chartdata),'chartlabels':json.dumps(chartlabels),'first_crime':first_crime,
    'second_crime':second_crime,'third_crime':third_crime,'location_address':location_address,'first_crime_no':first_crime_no,
    'second_crime_no':second_crime_no,'third_crime_no':third_crime_no,'total_crimes_percent':total_crimes_percent,'avg_time_diff':json.dumps(avg_time_diff),
    'region':region,'first_crime_percent':first_crime_percent,'second_crime_percent':second_crime_percent,
    'third_crime_percent':third_crime_percent,'total_crimes':total_crimes,'vertical_dist':vertical_dist,'horizontal_dist':horizontal_dist})

def nypd_clustermodel(clusteroption):
    import pandas as pd
    import matplotlib.pyplot as mtp
    import numpy as np
    import os
    from io import BytesIO
    import base64
    import geopy.distance
    os.system('pip install geopy')
    os.system('pip install geopandas')
    from geopy.geocoders import Nominatim
    import geopandas as gpd
    nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
    df_radio_code=pd.read_csv("datasets/radio_code_df.csv")
    df_radio_code.set_index("Unnamed: 0",inplace=True)
    if clusteroption=="one":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Manhattan'])
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Bronx'])
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='black', edgecolor='white', ax=ax)
        mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1],s=0.1,c = 'blue')
        mtp.annotate('MANHATTAN',xy=(-74.0,40.75),weight='bold',fontsize=12,color='White')
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.annotate('BRONX',xy=(-73.90,40.85),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close() 
        region = 1
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(0)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="two":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Brooklyn'])
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='black', edgecolor='white', ax=ax)
        mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 0.1, c = 'green')
        mtp.annotate('BROOKLYN',xy=(-74.00,40.63),weight='bold',fontsize=12,color='White')
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 2
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(1)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="three":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='black', edgecolor='white', ax=ax)
        mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 0.1, c = 'red')
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 3
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(2)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="four":
        nyc_new=nyc[nyc['BoroName']=='Bronx']
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 0.1, c = 'cyan') 
        mtp.annotate('BRONX',xy=(-73.90,40.85),weight='bold',fontsize=12,color='WHITE')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 4
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(3)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="five":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Manhattan'])
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Brooklyn'])
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 0.1, c = 'magenta') 
        mtp.annotate('MANHATTAN',xy=(-74.0,40.75),weight='bold',fontsize=12,color='White')
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.annotate('BROOKLYN',xy=(-74.00,40.63),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 5
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(4)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="six":
        nyc_new=nyc[nyc['BoroName']=='Staten Island']
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 5, 0], x[y_predict == 5, 1], s = 2, c = 'purple')
        mtp.annotate('STATEN ISLAND',xy=(-74.19,40.6),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 6
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(5)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
         
        
    elif clusteroption=="seven":
        nyc_new=nyc[nyc['BoroName']=='Brooklyn']
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 6, 0], x[y_predict == 6, 1], s = 2, c = 'pink')
        mtp.annotate('BROOKLYN',xy=(-73.95,40.655),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 7
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(6)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
       
    elif clusteroption=="eight":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 7, 0], x[y_predict == 7, 1], s = 0.1, c = 'orange')
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 8
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(7)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    elif clusteroption=="nine":
        nyc_new=nyc[nyc['BoroName']=='Queens']
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 8, 0], x[y_predict == 8, 1], s = 2, c = 'yellowgreen')   
        mtp.annotate('QUEENS',xy=(-73.85,40.70),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 9
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(8)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        lat_max = temp["Latitude"].max()
        lat_min = temp["Latitude"].min()
        long_max = temp["Longitude"].max()
        long_min = temp["Longitude"].min()
        lat_mid = (lat_max+lat_min)/2
        long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(lat_mid) + "," + str(long_mid))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
        
    else:
        nyc_new=nyc[nyc['BoroName']=='Bronx']
        nyc_new=nyc_new.append(nyc[nyc['BoroName']=='Manhattan'])
        import numpy as np
        df_wm = nyc_new.to_crs(epsg=4326)
        fig,ax = mtp.subplots(1,1, figsize=(10,10))
        base = df_wm.plot(color='BLACK', edgecolor='WHITE', ax=ax)
        mtp.scatter(x[y_predict == 9, 0], x[y_predict == 9, 1], s = 0.1, c = 'brown')
        mtp.annotate('MANHATTAN',xy=(-74.0,40.75),weight='bold',fontsize=12,color='White',rotation=40)
        mtp.annotate('BRONX',xy=(-73.90,40.85),weight='bold',fontsize=12,color='White')
        mtp.axis('off')
        buffer = BytesIO()
        mtp.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        cluster_graph = base64.b64encode(image_png)
        cluster_graph = cluster_graph.decode('utf-8')
        buffer.close()
        region = 10
        #Find the most occuring radio codes (top 3) in each cluster.
        temp = df_groupby_labels.get_group(9)
        vc = temp["RADIO_CODE"].value_counts()[:3]
        first_crime = df_radio_code.loc[vc.index[0]]['TYP_DESC']
        second_crime = df_radio_code.loc[vc.index[1]]['TYP_DESC']
        third_crime = df_radio_code.loc[vc.index[2]]['TYP_DESC']
        first_crime_no=vc[0]
        second_crime_no=vc[1]
        third_crime_no=vc[2]
        #Assign an area name to each cluster
        geolocator = Nominatim(user_agent="geoapiExercises")
        # lat_max = temp["Latitude"].max()
        # lat_min = temp["Latitude"].min()
        # long_max = temp["Longitude"].max()
        # long_min = temp["Longitude"].min()
        # lat_mid = (lat_max+lat_min)/2
        # long_mid = (long_max+long_min)/2
        location_address = geolocator.reverse(str(kmeans_cluster_centers_1[9]) + "," + str(kmeans_cluster_centers_0[9]))
        #percent of crimes in region
        total_crimes_percent=int((len(temp)/total_crimes)*100)
        #total crimes in region
        total_crimes_region=len(temp)
        #percent of top 3 crimes in region
        first_crime_percent = int((first_crime_no/total_crimes_region)*100)
        second_crime_percent = int((second_crime_no/total_crimes_region)*100)
        third_crime_percent = int((third_crime_no/total_crimes_region)*100)
        #finding horizontal and vertical distance
        vertical_lat_1=max(temp['Latitude'])
        vertical_long_1=float(temp[temp['Latitude']==max(temp['Latitude'])].head(1)['Longitude'])
        vertical_lat_2=min(temp['Latitude'])
        vertical_long_2=float(temp[temp['Latitude']==min(temp['Latitude'])].head(1)['Longitude'])
        vertical_coords_1=(vertical_lat_1,vertical_long_1)
        vertical_coords_2=(vertical_lat_2,vertical_long_2)
        vertical_dist=round(geopy.distance.geodesic(vertical_coords_1, vertical_coords_2).km,2)
        horizontal_lat_1=float(temp[temp['Longitude']==min(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_1=min(temp['Longitude'])
        horizontal_lat_2=float(temp[temp['Longitude']==max(temp['Longitude'])].head(1)['Latitude'])
        horizontal_long_2=max(temp['Longitude'])
        horizontal_coords_1=(horizontal_lat_1,horizontal_long_1)
        horizontal_coords_2=(horizontal_lat_2,horizontal_long_2)
        horizontal_dist=round(geopy.distance.geodesic(horizontal_coords_1, horizontal_coords_2).km,2)
    
    return([cluster_graph,first_crime,second_crime,third_crime,location_address,first_crime_no,second_crime_no,third_crime_no,total_crimes_percent,region,
    first_crime_percent, second_crime_percent, third_crime_percent, total_crimes_region, vertical_dist, horizontal_dist])



def nypdtype(request):
    # import os
    # os.system('pip install geopandas')
    # import pandas as pd
    # import geopandas as gpd
    # from shapely import wkt
    # import seaborn as sns
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import io
    # import urllib, base64
    # from io import BytesIO
    # df1=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/df_after_preprocessing2.csv")
    # df2=pd.read_csv("/Users/kunjgala/Desktop/finalyearproject/model/NYPD_Calls_for_Service__Year_to_Date.csv")
    # global df
    # df = pd.merge(df1, df2, on='CAD_EVNT_ID', how='left')
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    # df.drop(['Latitude_y','Longitude_y','RADIO_CODE_y'], axis=1, inplace=True)
    # df.rename(columns = {'Latitude_x':'Latitude','Longitude_x':'Longitude','RADIO_CODE_x':'RADIO_CODE'}, inplace = True)
    # df.to_csv('crime_type_csv.csv')

    radio_code_list=["22Q2","29H1","39T1","20R","53D","24Q2","50G2","10S2","33","29Q1"]
    for i in range(0,len(radio_code_list)):
        output=mlmodel(radio_code_list[i])
        if i==0:
            plot1=output[0]
            max1=output[1]
            min1=output[2]
            maxpercent1=output[3]
            minpercent1=output[4]
            count1=output[5]
        elif i==1:
            plot2=output[0]
            max2=output[1]
            min2=output[2]
            maxpercent2=output[3]
            minpercent2=output[4]
            count2=output[5]
        elif i==2:
            plot3=output[0]
            max3=output[1]
            min3=output[2]
            maxpercent3=output[3]
            minpercent3=output[4]
            count3=output[5]
        elif i==3:
            plot4=output[0]
            max4=output[1]
            min4=output[2]
            maxpercent4=output[3]
            minpercent4=output[4]
            count4=output[5]
        elif i==4:
            plot5=output[0]
            max5=output[1]
            min5=output[2]
            maxpercent5=output[3]
            minpercent5=output[4]
            count5=output[5]
        elif i==5:
            plot6=output[0]
            max6=output[1]
            min6=output[2]
            maxpercent6=output[3]
            minpercent6=output[4]
            count6=output[5]
        elif i==6:
            plot7=output[0]
            max7=output[1]
            min7=output[2]
            maxpercent7=output[3]
            minpercent7=output[4]
            count7=output[5]
        elif i==7:
            plot8=output[0]
            max8=output[1]
            min8=output[2]
            maxpercent8=output[3]
            minpercent8=output[4]
            count8=output[5]
        elif i==8:
            plot9=output[0]
            max9=output[1]
            min9=output[2]
            maxpercent9=output[3]
            minpercent9=output[4]
            count9=output[5]
        else:
            plot10=output[0]
            max10=output[1]
            min10=output[2]
            maxpercent10=output[3]
            minpercent10=output[4]
            count10=output[5]

    nypd_crime_page ={
        'plot1':plot1,
        'max1': max1,
        'min1':min1,
        'maxpercent1': maxpercent1,
        'minpercent1': minpercent1,
        'count1':count1,

        'plot2':plot2,
        'max2': max2,
        'min2':min2,
        'maxpercent2': maxpercent2,
        'minpercent2': minpercent2,
        'count2':count2,

        'plot3':plot3,
        'max3': max3,
        'min3':min3,
        'maxpercent3': maxpercent3,
        'minpercent3': minpercent3,
        'count3':count3,

        'plot4':plot4,
        'max4': max4,
        'min4':min4,
        'maxpercent4': maxpercent4,
        'minpercent4': minpercent4,
        'count4':count4,

        'plot5':plot5,
        'max5': max5,
        'min5':min5,
        'maxpercent5': maxpercent5,
        'minpercent5': minpercent5,
        'count5':count5,

        'plot6':plot6,
        'max6': max6,
        'min6':min6,
        'maxpercent6': maxpercent6,
        'minpercent6': minpercent6,
        'count6':count6,

        'plot7':plot7,
        'max7': max7,
        'min7':min7,
        'maxpercent7': maxpercent7,
        'minpercent7': minpercent7,
        'count7':count7,

        'plot8':plot8,
        'max8': max8,
        'min8':min8,
        'maxpercent8': maxpercent8,
        'minpercent8': minpercent8,
        'count8':count8,

        'plot9':plot9,
        'max9': max9,
        'min9':min9,
        'maxpercent9': maxpercent9,
        'minpercent10': minpercent9,
        'count9':count9,

        'plot10':plot10,
        'max10': max10,
        'min10':min10,
        'maxpercent10': maxpercent10,
        'minpercent10': minpercent10,
        'count10':count10,

    }

    

    return render(request, 'dashboard/nypdtype.html',nypd_crime_page)



def mlmodel(radio_code):
    import matplotlib.pyplot as plt
    import urllib, base64
    from io import BytesIO
    import pandas as pd
    crime_type_csv = pd.read_csv("datasets/crime_type_csv.csv")
    radio=radio_code
    df_radio = crime_type_csv[crime_type_csv['RADIO_CODE']==(radio)]
    x = df_radio[['CAD_EVNT_ID','Latitude','Longitude','BORO_NM']]
    plt.switch_backend('AGG')
    plt.figure(figsize=(5,5))
    plt.title('Hello')
    import numpy as np
    import geopandas as gpd
    from shapely import wkt
    nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
    borough_count = x.groupby('BORO_NM').agg('count').reset_index()
    maxi=list(borough_count[max(borough_count['CAD_EVNT_ID'])==borough_count['CAD_EVNT_ID']]['BORO_NM'])[0]
    counti=x.shape[0]
    mini=list(borough_count[min(borough_count['CAD_EVNT_ID'])==borough_count['CAD_EVNT_ID']]['BORO_NM'])[0]
    maxpercenti=round(int(borough_count[borough_count['BORO_NM']==maxi]['CAD_EVNT_ID'])/counti*100)
    minpercenti=round(int(borough_count[borough_count['BORO_NM']==mini]['CAD_EVNT_ID'])/counti*100)
    nyc.rename(columns={'BoroName':'BORO_NM'}, inplace=True)
    for i in nyc.index:
        nyc.at[i, "BORO_NM"] = nyc.at[i, "BORO_NM"].upper()
    bc_geo = nyc.merge(borough_count, on='BORO_NM')
    fig,ax = plt.subplots(1,1, figsize=(7,5))
    bc_geo.plot(column='CAD_EVNT_ID', cmap='viridis_r', alpha=.5, ax=ax, legend=True)
    bc_geo.apply(lambda x: ax.annotate(text='%s \n %.f'%(x.BORO_NM,x.CAD_EVNT_ID ), color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
    # Number of Crimes by NYC Borough for Crime Type "+ borough
    plt.axis('off')
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    ploti = base64.b64encode(image_png)
    ploti = ploti.decode('utf-8')
    buffer.close()
    return([ploti,maxi,mini,maxpercenti,minpercenti,counti])


def twitter_original(request):
    import os
    os.system('pip install tweepy')
    import tweepy
    import pandas as pd
    import re
    # from wordcloud import WordCloud
    consumerKey = '9Ezv1ZK0ZvHbhGuIVsx8yE1j1'
    consumerSecret = 'dUL12PXcwvBSha8zXxiopuhb9SIgK4jXFYFF2rnejz8nAZO3C6'
    accessToken = '1490633689952231430-gDuJEQfUI3vrTTQ2Jcbub1dOzOKmOc'
    accessTokenSecret = 'ZkySwYPqt8t7l4f4jFKYySx9lXEz90kGNTGxBBn0JvJ2A'

    authenticate=tweepy.OAuthHandler(consumerKey, consumerSecret)
    authenticate.set_access_token(accessToken, accessTokenSecret)

    api = tweepy.API(authenticate, wait_on_rate_limit=True)
    posts = api.user_timeline(screen_name="NYPDnews", count=500, lang="en", tweet_mode="extended")

    #creating a dataset
    df_1 = pd.DataFrame( [tweet.full_text for tweet in posts], columns=['Tweets'] )


    #READING 2ND DATA SOURCE

    posts_2 = api.user_timeline(screen_name="NYPDTips", count=500, lang="en", tweet_mode="extended")


    #creating a dataset
    df_2 = pd.DataFrame( [tweet.full_text for tweet in posts_2], columns=['Tweets'] )


    df = df_1.append(df_2, ignore_index=True)

    df_original = df_1.append(df_2, ignore_index=True)

    # delete later
    # df = pd.read_csv('df_original_highlighted_csv.csv')
    # df_original = pd.read_csv('df_original_highlighted_csv.csv')


    #cleaning the tweets
    #removing unwanted characters

    def clean(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'[^\w\s\/]', '', text)
        return text


    df["Tweets"] = df["Tweets"].apply(clean)
    df.head(10)

    #converting all tweets to lower case
    df["Tweets"] = df["Tweets"].str.lower()

    #tokenization
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    def tokenize(text):
        return nltk.word_tokenize(text)

    df["Tokenized_Tweets"] = df["Tweets"].apply(tokenize)


    #removal of stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')


    def remove_stopwords(text_arr):
        for word in stop_words:
            while(word in text_arr):
                text_arr.remove(word)
        return text_arr

    df["Without_Stopwords_Tweets"] = df["Tokenized_Tweets"].apply(remove_stopwords)
    df.drop(columns='Tokenized_Tweets',axis=1,inplace=True)
    inverted_indexxx = []
    df_inverted_index = pd.DataFrame(inverted_indexxx,columns=["Word","Tweet_Index","Count"])
    # creating an inverted index
    def inverted_index(text_arr,doc_index):
        for word in text_arr:
            i = df_inverted_index.index[df_inverted_index['Word'] == word].to_list()
            if i:
                df_inverted_index.iloc[i[0]]["Tweet_Index"].append(doc_index)
                df_inverted_index.loc[i[0]]["Count"] = int(df_inverted_index.loc[i[0]]["Count"])+1
            else:
                df_inverted_index.loc[len(df_inverted_index.index)] = [word, [doc_index], 1]



    i = 0
    for tweet_arr in df["Without_Stopwords_Tweets"]:
        inverted_index(tweet_arr,i)
        i+=1

    df_inverted_index.to_csv("datasets/inverted_index_tweets.csv")


    docs = df["Tweets"]
    query = "wanted or robbery or assault or gun or shot or wantedfor or forcibly or stabbed or firearm or arrested or punched or burglary or perpetrator or wantedrobbery or reckless or endangerment or arrest or struck or grabbed or guns or firearms or touching or loaded or dispute or crime or assaulted or forcible or killed or rifles or illegal or knife or buttocks or rape or fleeing or broke or lewdness or criminal or gang or pushed or robberies or handguns or hit or kicked or homicide or shotguns or wantedassault or larceny or stole or missing or murder or violence or ghost or harassment or fire or burglaries or violent or manslaughter or apprehension or stolen or assaults or murders or sharp or wantedassaults or shooting or chokehold or raped"

    # building Boolean Model
    #creating a bitmap according to the given query
    bitmap = []
    words_all = query.split(' ')
    words_required = []
    words_connectors = []

    for i in range(0,len(words_all)):
        if i%2==0:
            words_required.append(words_all[i])
        else:
            words_connectors.append(words_all[i])

    query_len = len(words_required)

    for tweet in docs:
        temp = []
        for word in words_required:
            if word in tweet:
                temp.append(1)
            else:
                temp.append(0)
        bitmap.append(temp)


    bitmap_results = []
    for doc in bitmap:
        for i in range(0,query_len-1):
            if i==0:
                if words_connectors[i]=="and":
                    result = doc[i] & doc[i+1]
                else:
                    result = doc[i] | doc[i+1]
            elif words_connectors[i]=="and":
                result = result & doc[i+1]
            else:
                result = result | doc[i+1]
        bitmap_results.append(result)





    df_crimes_edited = df.copy(deep=True)
    for i in range(0,len(bitmap_results)):
        if bitmap_results[i]==0:
             df_crimes_edited.drop(index=i,axis=0,inplace=True)


    df_crimes = df_original.copy(deep=True)
    for i in range(0,len(bitmap_results)):
        if bitmap_results[i]==0:
            df_crimes.drop(index=i,axis=0,inplace=True)




    os.system('!pip install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.1.0/xx_ent_wiki_sm-3.1.0.tar.gz')
    print("Installation complete")

    # approach




    os.system('pip3 install spacy')
    os.system('python3 -m spacy download en')
    os.system('python3 -m spacy download xx_ent_wiki_sm')
    os.system('pip3 install -U pip setuptools wheel')
    os.system('pip3 install -U spacy')
    os.system('python3 -m spacy download xx_ent_wiki_sm')
    os.system('pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz')
    import spacy
    from spacy import displacy
    nlp_wk = spacy.load("xx_ent_wiki_sm")

    def extract_location(tweet):
        doc=nlp_wk(tweet)
        return [ent.text for ent in doc.ents if ent.label_ in ['LOC']]

    df_crimes["Location"] = df_crimes["Tweets"].apply(extract_location)


    # Import the required library
    from geopy.geocoders import Nominatim

    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")
    #geocoding
    def get_geocode(loc):
        coor = []
        loc=str(loc).replace("'","").replace("[","").replace("]","")
        try:
            location = geolocator.geocode(loc)
        except:
            location=None
        
        if location!=None:
            coor.append(location.latitude)
            coor.append(location.longitude)
        return coor

    # 2 seperate columns- LATITUDE
    # Import the required library
    from geopy.geocoders import Nominatim

    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")
    #geocoding
    def get_geocode_lat(loc):
        loc=str(loc).replace("'","").replace("[","").replace("]","")
        try:
            location = geolocator.geocode(loc)
        except:
            location=None
        
        if location!=None:
            return location.latitude
        return None

    # 2 seperate columns- LONGITUDE
    # Import the required library
    from geopy.geocoders import Nominatim

    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")
    #geocoding
    def get_geocode_lon(loc):
        loc=str(loc).replace("'","").replace("[","").replace("]","")
        try:
            location = geolocator.geocode(loc)
        except:
            location=None
        
        if location!=None:
            return location.longitude
        return None

    df_crimes["Latitude"] = df_crimes["Location"].apply(get_geocode_lat)
    df_crimes["Longitude"] = df_crimes["Location"].apply(get_geocode_lon)


    #dropping the rows which do not have coordinate info
    print("Number of rows having coordinates:",len(df_crimes)-df_crimes["Latitude"].isnull().sum())
    df_crimes_coor = df_crimes.dropna(axis=0)
    df_crimes_coor.shape

    # long: -74 to -73.7
    # lat: 40.5 to 40.9
    #dropping rows which are outliers acc to geo coordinates

    df_crimes_coor = df_crimes_coor.loc[(df_crimes_coor['Longitude'] >= -75)]
    df_crimes_coor = df_crimes_coor.loc[(df_crimes_coor['Longitude'] <= -73.6)]
    df_crimes_coor = df_crimes_coor.loc[(df_crimes_coor['Latitude'] <= 42)]
    df_crimes_coor = df_crimes_coor.loc[(df_crimes_coor['Latitude'] >= 40)]


    #df_crimes = pd.read_csv('extracted coordinates.csv')
    #df_crimes
    df_crimes_coor.to_csv("datasets/extracted coordinates filtered.csv")




    os.system('pip install folium')
    import folium
    from folium.plugins import FastMarkerCluster

    pointIcon_url = "http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"
    icon = folium.features.CustomIcon(pointIcon_url, icon_size=(15, 15))

    folium_map = folium.Map(location=[40.7,-74],
    zoom_start=11,
    tiles='CartoDB dark_matter',width=500,height=500)
    FastMarkerCluster(data=list(zip(df_crimes_coor["Latitude"].values, df_crimes_coor["Longitude"].values))).add_to(folium_map)
    folium.LayerControl().add_to(folium_map)



    # saving the map as a HTML file
    folium_map.save(outfile= "folium_output_original.html")

    # getting borough-wise stats
    brooklyn=0
    manhattan=0
    bronx=0
    statenIsland=0
    queens=0
    print(df_crimes_coor.shape)
    for i in df_crimes_coor['Location']:
        i=[x.lower() for x in i]
        if 'brooklyn' in i:
            brooklyn=brooklyn+1
        if 'manhattan' in i:
            manhattan=manhattan+1
        if 'bronx' in i:
            bronx=bronx+1
        if 'queens' in i:
            queens=queens+1
        if 'staten island' in i:
            statenIsland=statenIsland+1
    
    print("No. of crimes in Brooklyn: ",brooklyn)
    print("No. of crimes in Bronx: ",bronx)
    print("No. of crimes in Manhattan: ",manhattan)
    print("No. of crimes in Queens: ",queens)
    print("No. of crimes in Staten Island: ",statenIsland)

 


    return render(request, 'dashboard/twitter.html',{'brooklyn':brooklyn,'bronx':bronx,'manhattan':manhattan,'queens':queens,'statenIsland':statenIsland})


def twitter(request):
    return render(request, 'dashboard/twitter.html')




from .charts import outputCharts

def newsfeed(request):
    byDays, borough_summary = outputCharts.get_time_series_data()
    context = {}
    context['dates'] = byDays.keys()
    context['values'] = byDays.values()
    context['borough_names'] = borough_summary['borough']
    context['borough_count'] = borough_summary['count']
    context['coordinates'] = []
    data = outputCharts.get_news_folium_data()
    for i in range(len(data['Latitude'])):
        context['coordinates'].append([data['Latitude'][i], data['Longitude'][i]])
    
    return render(request, 'dashboard/newsfeed.html',context)


def news_folium_map(request):
    data = outputCharts.get_news_folium_data()
    print(data)

    context = {}
    context['coordinates'] = []
    for i in range(len(data['Latitude'])):
        context['coordinates'].append([data['Latitude'][i], data['Longitude'][i]])
    # context['lat'] = data['Latitude']
    # context['lon'] = data['Longitude']
    return render(request, 'partials/folium_output_news.html',context)