from django.urls import path

from dashboard.chartView import EditorChartView
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('nypdlocation/', views.nypdlocation, name='dashboard-nypdlocation'),
    path('nypdtype/', views.nypdtype, name='dashboard-nypdtype'),
    path('twitter/', views.twitter, name='dashboard-twitter'),
    path('newsfeed/', views.newsfeed, name='dashboard-newsfeed'),
    path('plot/', views.plot, name='plot'),
    path('stat/', EditorChartView.as_view()),
   
   
    
    
]