from django.contrib import admin
from .models import Company

class CompanyAdmin(admin.ModelAdmin):
    list_display = ['start_year', 'founder', 'revenue']

admin.site.register(Company, CompanyAdmin)
