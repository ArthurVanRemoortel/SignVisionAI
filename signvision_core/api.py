from rest_framework import viewsets

import signvision_core.serializers as serializers
from signvision_core.models import Gesture, Word, SignLanguage, Country


class GesturesViewSet(viewsets.ModelViewSet):
    queryset = Gesture.objects.all()
    serializer_class = serializers.GestureSerializer


class WordViewSet(viewsets.ModelViewSet):
    queryset = Word.objects.all()
    serializer_class = serializers.WordSerializer


class SignLanguageViewSet(viewsets.ModelViewSet):
    queryset = SignLanguage.objects.all()
    serializer_class = serializers.SignLanguageSerializer


class CountryViewSet(viewsets.ModelViewSet):
    queryset = Country.objects.all()
    serializer_class = serializers.CountrySerializer

