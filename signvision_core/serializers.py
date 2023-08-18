from rest_framework import serializers
from signvision_core.models import Gesture, Word, SignLanguage, Country


class GestureSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Gesture
        fields = ['id', 'word', 'hands']


class WordSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Word
        fields = ['id', 'word']


class SignLanguageSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SignLanguage
        fields = ['id', 'name', 'abbreviation', 'description']


class CountrySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Country
        fields = ['id', 'name']