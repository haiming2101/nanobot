from django.contrib.contenttypes.views import shortcut
from django.urls import path, re_path

from nano.comments import views


urlpatterns = [
    re_path(r'^cr/(\d+)/(.+)/$', shortcut, name='comments-url-redirect'),
    path('post',     views.post_comment, name='comments-post-comment'),
    path('',         views.list_comments, name='comments-list-comments'),
]
