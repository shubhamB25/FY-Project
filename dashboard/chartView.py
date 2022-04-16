from operator import imod
from django.shortcuts import render
from django.views.generic import TemplateView
import pandas as pd
from .model import outputCharts
from django.core.cache import cache

# Creating views
class EditorChartView(TemplateView):
    template_name = 'editors/chart.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        borough_data = outputCharts.getBoroughStats()
        data = {"borough": borough_data['BORO_NM'], "count":borough_data['CAD_EVNT_ID']}
        # cache.set('key',borough_data)
        # data = {"city1":200000, "city2":300000,"city3": 30000}
        context["qs"] = data
        print(context['qs'])
        return context