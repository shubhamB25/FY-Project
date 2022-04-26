from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('nypdlocation/', views.nypdlocation, name='dashboard-nypdlocation'),
    path('nypdtype/', views.nypdtype, name='dashboard-nypdtype'),
    path('twitter/', views.twitter, name='dashboard-twitter'),
    path('twitteroriginal/', views.twitter_original, name='twitter'),
    path('newsfeed/', views.newsfeed, name='dashboard-newsfeed'),
    path('newsfeed/map', views.news_folium_map, name='dashboard-newsfeed-map'),
    path('cluster/', views.cluster, name='cluster'),
   
   
    
    
]