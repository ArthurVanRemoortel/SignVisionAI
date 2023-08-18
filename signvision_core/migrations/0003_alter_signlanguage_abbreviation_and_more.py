# Generated by Django 4.2.4 on 2023-08-12 22:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('signvision_core', '0002_country_sign_languages_gesture_hands_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='signlanguage',
            name='abbreviation',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='signlanguage',
            name='description',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='word',
            name='word',
            field=models.CharField(max_length=255),
        ),
    ]
