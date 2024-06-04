from django.contrib import admin
from .models import Company

class CompanyAdmin(admin.ModelAdmin):
    list_display = ('username', 'start_year', 'revenue', 'founder', 'investment_round')
    search_fields = ('username__username', 'start_year', 'founder', 'investment_round')
    list_filter = ('start_year', 'investment_round')

admin.site.register(Company, CompanyAdmin)
