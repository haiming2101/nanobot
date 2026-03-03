from django.urls import include, path, re_path

from nano.privmsg import views


urlpatterns = [
    re_path(r'^(?P<msgid>[1-9][0-9]*)/', include([
        path('archive', views.move_to_archive, name='archive_pm'),
        path('delete', views.delete, name='delete_pm'),
    ])),
    #re_path(r'^(?:(?P<action>(archive|sent))/?)?$', views.show_pms, name='show_pms'),
    path('add', views.add_pm, name='add_pm'),
    path('archive/', views.show_pm_archived, name='show_archived_pms'),
    path('sent/', views.show_pm_sent, name='show_sent_pms'),
    path('', views.show_pm_received, name='show_pms'),
    #re_path(r'^$', views.show_pms, {u'action': u'received'}, name='show_pms'),
]
