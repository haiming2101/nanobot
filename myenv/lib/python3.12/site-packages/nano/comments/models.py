from django.conf import settings
from django.db import models
from django.utils.encoding import force_str
from django.utils.timezone import now as tznow

from nano.tools.models import GenericForeignKeyAbstractModel
from nano.tools.models import UnorderedTreeManager
from nano.tools.models import UnorderedTreeMixin
from nano.tools.models import get_manager


COMMENT_MAX_LENGTH = getattr(settings,'COMMENT_MAX_LENGTH',3000)


class Comment(GenericForeignKeyAbstractModel, UnorderedTreeMixin):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="%(class)s_comments",
    )
    comment = models.TextField(max_length=COMMENT_MAX_LENGTH)
    comment_xhtml = models.TextField(editable=False)
    added = models.DateTimeField(default=tznow)
    is_visible = models.BooleanField(default=True)
    is_scrambled = models.BooleanField(default=False)

    objects = models.Manager()
    tree = UnorderedTreeManager()

    class Meta:
        db_table = "nano_comments_comment"
        ordering = ('added',)
        get_latest_by = 'added'

    def __str__(self):
        comment = self.comment
        if len(comment) > 50:
            comment = self.comment[:49]+'...'
        return f"{self.user}: {comment}"

    def get_content_object_url(self):
        return self.content_object.get_absolute_url() or ''

    def get_absolute_url(self, anchor_pattern="#c%(id)s"):
        content_url = self.get_content_object_url()
        anchor = anchor_pattern % self.__dict__
        if content_url:
            return content_url + anchor
        return anchor

    def roots(self):
        manager = get_manager(self)
        tree = manager.filter(part_of__isnull=True)
        return tree.filter(content_type=self.content_type, object_pk=self.object_pk)

    def get_path(self):
        manager = get_manager(self)
        return [manager.get(id=p).filter(content_type=self.content_type, object_pk=self.object_pk)
                for p in force_str(self.path).split(self._sep) if p]

    def descendants(self):
        manager = get_manager(self)
        tree = manager.filter(path__startswith=self.path).exclude(id=self.id)
        return tree.filter(content_type=self.content_type, object_pk=self.object_pk)
