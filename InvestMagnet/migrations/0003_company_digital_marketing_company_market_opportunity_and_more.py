# Generated by Django 5.0.6 on 2024-06-04 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('InvestMagnet', '0002_remove_company_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='company',
            name='digital_marketing',
            field=models.JSONField(default=[{'facebook': '5%', 'instagram': '20%', 'other_source': '30%', 'website': '35%', 'youtube': '10%'}]),
        ),
        migrations.AddField(
            model_name='company',
            name='market_opportunity',
            field=models.TextField(default='Market opportunity for the company'),
        ),
        migrations.AddField(
            model_name='company',
            name='problem_statement',
            field=models.TextField(default='Your problem statement'),
        ),
        migrations.AddField(
            model_name='company',
            name='solution',
            field=models.TextField(default='Solution for that problem statement'),
        ),
        migrations.AddField(
            model_name='company',
            name='team',
            field=models.IntegerField(default=10),
        ),
        migrations.AddField(
            model_name='company',
            name='yearwise_growth',
            field=models.JSONField(default=[{'growth': '20%', 'year': '2020'}, {'growth': '5%', 'year': '2021'}, {'growth': '2%', 'year': '2022'}, {'growth': '4%', 'year': '2023'}]),
        ),
        migrations.AddField(
            model_name='company',
            name='yearwise_revenue',
            field=models.JSONField(default=[{'revenue': '500000', 'year': '2020'}, {'revenue': '50034', 'year': '2021'}, {'revenue': '500000', 'year': '2022'}, {'revenue': '40000', 'year': '2023'}]),
        ),
        migrations.AlterField(
            model_name='company',
            name='description',
            field=models.TextField(default='A description of the company'),
        ),
        migrations.AlterField(
            model_name='company',
            name='founder',
            field=models.CharField(default='John Doe', max_length=255),
        ),
        migrations.AlterField(
            model_name='company',
            name='investment_round',
            field=models.CharField(default='Series A', max_length=255),
        ),
        migrations.AlterField(
            model_name='company',
            name='revenue',
            field=models.CharField(default='1000000', max_length=255),
        ),
        migrations.AlterField(
            model_name='company',
            name='start_year',
            field=models.CharField(default='2020', max_length=4),
        ),
    ]
