"""
Wrapper for loading templates from the filesystem.
"""

from django.apps import apps
from django.conf import settings
from django.template import TemplateDoesNotExist
from django.template.base import Origin
from django.utils._os import safe_join

from django.template.loaders.base import Loader


class ChunkLoader(Loader):
    is_usable = True

    def _get_contents(self, template_name):
        chunk_model = apps.get_model('chunk', 'Chunk')
        try:
            chunk = chunk_model.objects.get(slug=template_name)
            return chunk.content
        except chunk_model.DoesNotExist:
            error_msg = "Couldn't find a chunk named %s" % template_name
            raise TemplateDoesNotExist(error_msg)

    # Django 1.11
    def load_template_source(self, template_name, template_dirs=None):
        return self._get_contents(template_name), template_name

    # Django 2.0+
    def get_contents(self, origin):
        return self._get_contents(origin.name)

    def get_template_sources(self, template_name):
        return [Origin(template_name, template_name, loader=self)]
