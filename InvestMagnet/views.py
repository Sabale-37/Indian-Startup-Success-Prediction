from django.shortcuts import render, redirect,get_object_or_404
from django.http import HttpResponse
from .models import Company


# Create your views here.
def company_list(request):
    companies = Company.objects.all()
    return render(request, 'company_list.html', {'companies': companies})

def company_detail(request, company_id):
    company = get_object_or_404(Company, id=company_id)
    return render(request, 'company_detail.html', {'company': company})


def add_company(request):
    if request.method == 'POST':
        username = request.user
        start_year = request.POST.get('start_year')
        description = request.POST.get('description')
        image = request.FILES.get('image')
        revenue = request.POST.get('revenue')
        founder = request.POST.get('founder')
        investment_round = request.POST.get('investment_round')

        company = Company(
            username = username,
            start_year=start_year,
            description=description,
            image=image,
            revenue=revenue,
            founder=founder,
            investment_round=investment_round
        )
        company.save()
        return render(request, 'add_company.html')
    
    return render(request, 'add_company.html')


def pitch(request):
    if request.method =="POST":
        username = request.POST.get('username')

    return render(request, 'pitch.html', {'username': username})
   