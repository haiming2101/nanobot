from django.contrib import admin

from nano.mark.models import MarkType, Mark

@admin.register(Mark)
class MarkAdmin(admin.ModelAdmin):
    model = Mark
    ordering = ('marked_at',)
    list_display = ('marktype', 'marked_by')
    list_filter = ('marktype',)

@admin.register(MarkType)
class MarkTypeAdmin(admin.ModelAdmin):
    model = MarkType
    ordering = ('name',)
    list_display = ('name',)
    list_filter = ('verify','permanent','hide')

