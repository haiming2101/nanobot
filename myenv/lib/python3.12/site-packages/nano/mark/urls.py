from django.urls import path

from nano.mark import views


urlpatterns = [
    path('toggle', views.toggle_mark, name='toggle_mark'),
]
