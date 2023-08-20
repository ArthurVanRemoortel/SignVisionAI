import json
from pprint import pprint

from django.db.models import Q
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.views import APIView

import signvision_core.serializers as serializers
from signvision_ai.classifiers.gesture_classifier import GestureClassifier
from signvision_ai.dataset import GestureDatasetEntry
from signvision_ai.models.gesture_model import GestureModel
from signvision_core.models import Gesture, Word, SignLanguage, Country
from django.utils.decorators import method_decorator


class ClassifierWrapper:
    _instances: {SignLanguage: GestureClassifier} = {}

    def __new__(cls, language: SignLanguage) -> GestureClassifier:
        key = language
        if key not in cls._instances:
            cls._instances[key] = GestureClassifier(language)
            model_config = cls._instances[key].gesture_model.model.get_config()
            if 'layers' in model_config:
                for layer_config in model_config['layers']:
                    if 'config' in layer_config and 'class_name' in layer_config:
                        if layer_config['class_name'] == 'Dense':  # Assuming you're using a Dense layer for classification
                            dense_config = layer_config['config']
                            if 'units' in dense_config and 'activation' in dense_config:
                                print("Classifier Labels:", dense_config['units'])
                                print("Activation:", dense_config['activation'])
        return cls._instances[key]

    @classmethod
    def preload_classifiers(cls):
        for language in SignLanguage.objects.prefetch_related('words').filter(abbreviation="VGT").all():
            if language.words.count() == 0:
                print(f"Skipping {language.abbreviation} because it has no words")
                continue
            # load_language = False
            # for word in language.words.all():
            #     for gesture in word.gestures.all():
            #         if gesture.dataset.count() >= 5:
            #             load_language = True
            #             break
            # if load_language:
            print(f"Preloading classifier for {language.abbreviation}")
            classifier = cls(language)


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


@method_decorator(csrf_exempt, name='dispatch')
class GestureClassificationAPIView(View):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        try:
            # Retrieve JSON data from the request body
            request_data = json.loads(request.body)
            language_param = request_data['language']
            gesture_data = request_data['gesture']

            language = SignLanguage.objects.filter(Q(abbreviation=language_param) | Q(name=language_param))
            if not language.exists():
                return JsonResponse({'error': f"Sing Language {language_param} does not exist."}, status=404)
            classifier: GestureClassifier = ClassifierWrapper(language.first())
            predict_entry = GestureDatasetEntry.from_json(gesture_data)
            result = classifier.classify(predict_entry)
            # predict_entry.prepare()
            pprint(result)
            response_data = {
                "predictions": {
                    gesture.word.word: confidence for gesture, confidence in result.items()
                }
            }
            # Return JSON response
            return JsonResponse(response_data, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)