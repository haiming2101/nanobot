
from django.contrib import admin

from nano.faq.models import *

@admin.register(QA)
class QAAdmin(admin.ModelAdmin):
    model = QA
    list_display = ('question', 'answer', 'last_modified')

