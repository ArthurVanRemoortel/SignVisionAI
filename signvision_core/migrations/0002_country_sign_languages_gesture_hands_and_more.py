# Generated by Django 4.2.4 on 2023-08-12 21:56

from django.db import migrations, models
import django.db.models.deletion
import signvision_core.models


class Migration(migrations.Migration):

    dependencies = [
        ('signvision_core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='country',
            name='sign_languages',
            field=models.ManyToManyField(related_name='countries', to='signvision_core.signlanguage'),
        ),
        migrations.AddField(
            model_name='gesture',
            name='hands',
            field=models.CharField(choices=[('left', 'LEFT'), ('right', 'RIGHT'), ('both', 'BOTH')], default=signvision_core.models.Hands['BOTH']),
        ),
        migrations.AddField(
            model_name='signlanguage',
            name='abbreviation',
            field=models.CharField(default=None, max_length=100),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='word',
            name='language',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='words', to='signvision_core.signlanguage'),
        ),
    ]
