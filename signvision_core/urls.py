from django.urls import path, include
from rest_framework import routers
from signvision_core import api

router = routers.DefaultRouter()
router.register(r'gestures', api.GesturesViewSet, 'gesture')
router.register(r'words', api.WordViewSet, 'word')
router.register(r'sign_languages', api.SignLanguageViewSet, 'sign_language')
router.register(r'countries', api.CountryViewSet, 'country')

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/classify_gesture/', api.GestureClassificationAPIView.as_view(), name='classify_gesture'),
]
