# Generated by Django 4.2.4 on 2023-08-13 16:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('signvision_core', '0008_gestureentry_mouth_gestureentry_entry_type'),
    ]

    operations = [
        migrations.RenameField(
            model_name='gestureentry',
            old_name='INDEX_BASE',
            new_name='LEFT_INDEX_BASE',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='INDEX_TIP',
            new_name='LEFT_INDEX_TIP',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='MIDDLE_BASE',
            new_name='LEFT_MIDDLE_BASE',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='MIDDLE_TIP',
            new_name='LEFT_MIDDLE_TIP',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='PINKY_BASE',
            new_name='LEFT_PINKY_BASE',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='PINKY_TIP',
            new_name='LEFT_PINKY_TIP',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='RING_BASE',
            new_name='LEFT_RING_BASE',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='RING_TIP',
            new_name='LEFT_RING_TIP',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='THUMB_BASE',
            new_name='LEFT_THUMB_BASE',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='THUMB_TIP',
            new_name='LEFT_THUMB_TIP',
        ),
        migrations.RenameField(
            model_name='gestureentry',
            old_name='WRIST',
            new_name='LEFT_WRIST',
        ),
    ]
