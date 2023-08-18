import os
import django
from pathlib import Path
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SignVisionSite.settings")
django.setup()
from signvision_core.models import Country, SignLanguage, Word, Gesture, Hands

BELGIUM, be_created = Country.objects.get_or_create(name="Belgium")
USA, usa_created = Country.objects.get_or_create(name="United States of America")
Canada, canada_created = Country.objects.get_or_create(name="Canada")

VGT, vgt_created = SignLanguage.objects.get_or_create(name="Vlaamse Gebarentaal", abbreviation="VGT", description="The language spoken in the Dutch speaking part of Belgium")
ASL, asl_created = SignLanguage.objects.get_or_create(name="Ameican Sign Language", abbreviation="ASL", description="The predominant sign language of Deaf communities in the United States of America and the Anglophone regions of Canada")

if be_created:
    BELGIUM.sign_languages.add(VGT)
    BELGIUM.save()

if usa_created:
    USA.sign_languages.add(ASL)
    USA.save()

if canada_created:
    Canada.sign_languages.add(ASL)
    Canada.save()






# for gesture_folder in Path('data/datasets/vgt-all').iterdir():
#     gesture_name, handedness_string = gesture_folder.name.split('_')
#     hand = Hands.BOTH
#     if handedness_string[0] == '1' and handedness_string[1] == '0':
#         hand = Hands.LEFT
#     elif handedness_string[0] == '0' and handedness_string[1] == '1':
#         hand = Hands.RIGHT
#     word, _ = Word.objects.get_or_create(word=gesture_name, language=VGT)
#     gesture = Gesture.objects.get_or_create(word=word, hands=hand)
