from django.shortcuts import render
from django.urls import path

from nano.activation.views import activate_key


urlpatterns = [
    path('activate',       activate_key, name='nano-activate-key'),
    path('activation_ok/', render,
                           {'template_name': 'nano/activation/activated.html'},
                           name='nano-activation-ok'),
]
