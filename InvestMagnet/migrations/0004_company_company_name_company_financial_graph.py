# Generated by Django 5.0.6 on 2024-06-04 18:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('InvestMagnet', '0003_company_digital_marketing_company_market_opportunity_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='company',
            name='company_name',
            field=models.CharField(default='amazon', max_length=30),
        ),
        migrations.AddField(
            model_name='company',
            name='financial_graph',
            field=models.JSONField(default=[{'new_income': '5000', 'total_sales': '10000', 'year': '2020'}, {'new_income': '7000', 'total_sales': '20000', 'year': '2021'}, {'new_income': '4000', 'total_sales': '22000', 'year': '2022'}, {'new_income': '3000', 'total_sales': '9000', 'year': '2023'}]),
        ),
    ]
