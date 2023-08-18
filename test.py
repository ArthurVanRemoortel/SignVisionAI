import math
import os
import time
from pprint import pprint
from tqdm import tqdm
import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SignVisionSite.settings")
django.setup()

from signvision_core.models import Word, GestureEntry, SignLanguage, Gesture, Hands
from signvision_ai.classifiers.gesture_classifier import GestureClassifier
from signvision_ai.dataset import HandFrame, Coordinate

if __name__ == "__main__":
    VGT = SignLanguage.objects.get(abbreviation="VGT")

    time0 = time.time()

    classifier = GestureClassifier(VGT)

    print("Completed in " + str(time.time() - time0) + " seconds")
