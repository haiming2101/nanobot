from django.urls import path

from nano.faq import views


urlpatterns = [
    path('', views.list_faqs),
]
