from functools import partial

from django.contrib import admin

from nano.blog.models import Entry
from nano.blog.settings import NANO_BLOG_TAGS


if NANO_BLOG_TAGS:
    from taggit.models import TaggedItem

    MAGICAL_TAGS = ('devel', 'pinned', 'news')


    class BlogHasTagFilter(admin.SimpleListFilter):
        title = 'having tags'
        parameter_name = 'has_any_tags'

        def lookups(self, request, model_admin):
            return (
                ('yes', 'Yes'),
                ('no', 'No'),
            )

        def queryset(self, request, queryset):
            if self.value() == 'yes':
                return queryset.exclude(tags=None)
            elif self.value() == 'no':
                return queryset.filter(tags=None)
            return queryset

    class BlogTagFilter(admin.SimpleListFilter):
        title = 'specific tags'
        parameter_name = 'has_tag'

        def _choices(self):
            tagged_items = TaggedItem.objects.filter(content_type__model=Entry._meta.model_name)
            blog_tags = tuple(tagged_items.values_list("tag__slug", "tag__name").distinct())
            return blog_tags

        def lookups(self, request, model_admin):
            return self._choices()

        def queryset(self, request, queryset):
            tags = [slug for (slug, _) in self._choices()]
            value = self.value()
            if value in tags:
                return queryset.filter(tags__slug=value)
            return queryset

    def _tag(modeladmin, request, queryset, tag):
        for entry in queryset:
            entry.tags.add(tag)

    def _untag(modeladmin, request, queryset, tag):
        for entry in queryset:
            entry.tags.remove(tag)

    def tagaction_factory():
        actions = []
        for tag in MAGICAL_TAGS:
            action = partial(_tag, tag=tag)
            action.short_description = f"Tag selected entries with '{tag}'"
            action.__name__ = f"tag_{tag}"
            actions.append(action)
            action = partial(_untag, tag=tag)
            action.short_description = f"Remove '{tag}'-tag from selected entries"
            action.__name__ = f"untag_{tag}"
            actions.append(action)
        return actions


@admin.register(Entry)
class EntryAdmin(admin.ModelAdmin):
    model = Entry
    list_display = ('headline', 'pub_date')
    search_fields = ('headline', 'pub_date')
    date_hierarchy = 'pub_date'

    if NANO_BLOG_TAGS:
        list_display = list_display + ("tag_list",)
        list_filter = [BlogHasTagFilter, BlogTagFilter]
        actions = tagaction_factory()

        def get_queryset(self, request):
            return super().get_queryset(request).prefetch_related('tags')

        def tag_list(self, obj):
            return ", ".join(o.name for o in obj.tags.all())
