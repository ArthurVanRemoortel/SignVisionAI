import os
import sys

from django.apps import AppConfig

class SignvisionCoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'signvision_core'

    def ready(self):
        server_version = os.environ.get('SERVER_SOFTWARE', '')

        if "gunicorn" not in server_version and "runserver" not in sys.argv:
            return

        if "runserver" in sys.argv and (
            os.environ.get("RUN_MAIN") != "true" and "--noreload" not in sys.argv
        ):
            return

        print("Server started:")
        from signvision_core.api import ClassifierWrapper
        ClassifierWrapper.preload_classifiers()
