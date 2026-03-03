from django.urls import path

from nano.badge import views


urlpatterns = [
    path('<int:pk>/',  views.show_badge, name='badge-detail'),
    path('',           views.list_badges, name='badge-list'),
]
