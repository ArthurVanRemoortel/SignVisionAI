# Generated by Django 4.2.4 on 2023-08-13 13:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('signvision_core', '0006_gestureentry_frame_count'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='gestureentry',
            table='gesture_entries',
        ),
    ]
