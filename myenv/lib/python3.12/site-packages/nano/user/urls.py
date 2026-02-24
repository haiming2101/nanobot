from django.shortcuts import render
from django.urls import include, path

from nano.user import views


signup_done_data = {'template_name': 'nano/user/signup_done.html'}

# 'project_name' should be a setting
password_reset_data = {'project_name': 'CALS'}

urlpatterns = [
    path('password/', include([
        path('reset/',   views.password_reset, password_reset_data, name='nano_user_password_reset'),
        path('change/',  views.password_change, name='nano_user_password_change'),
    ])),
    path('signup/', include([
        path('done/',      render, signup_done_data, name='nano_user_signup_done'),
        path('',           views.signup, name='nano_user_signup'),
    ])),
]
