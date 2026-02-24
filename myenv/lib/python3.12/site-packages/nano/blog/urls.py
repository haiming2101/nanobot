from django.urls import include, path, re_path

from nano.blog import views


urlpatterns = [
    re_path(r'^(?P<year>\d{4})/', include([
        re_path(r'^(?P<month>[01]\d)/(?P<day>[0123]\d)/$', views.list_entries_by_date),
        re_path(r'^(?P<month>[01]\d)/$', views.list_entries_by_year_and_month),
        path('', views.list_entries_by_year),
    ])),
    path('latest/',                views.list_latest_entries),
    path('today/',                 views.list_entries_for_today),
    path('',                       views.list_entries),
]
