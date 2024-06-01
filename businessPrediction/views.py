from django.shortcuts import render
from django.http import HttpResponse
from .startup_prediction import process_and_train_model

# Create your views here.

def prediction_business(request):
    return render(request, 'prediction.html')



def dashboard(request):
    if request.method == 'POST':
        # Get form data
        sector = request.POST.get('Sector')
        city = request.POST.get('City')
        startup_name = request.POST.get('Startup_Name')
        founded_year = int(request.POST.get('Founded_Year', 0))
        funding_amount = int(request.POST.get('Funding_Amount', 0))
        funding_rounds = int(request.POST.get('Funding_Rounds', 0))
        investor_count = int(request.POST.get('Investor_Count', 0))
        team_size = int(request.POST.get('Team_Size', 0))
        revenue = int(request.POST.get('Revenue', 0))
        profit_margin = float(request.POST.get('Profit_Margin', 0))
        customer_count = int(request.POST.get('Customer_Count', 0))
        growth_rate = float(request.POST.get('Growth_Rate', 0))
        founder_experience = int(request.POST.get('Founder_Experience', 0))
        market_size = int(request.POST.get('Market_Size', 0))

        # Now you have all the form data, you can use it as needed
        # For example, you can process it, save it to the database, etc.
        performance = process_and_train_model(
            sector=sector,
            city=city,
            startup_name=startup_name,
            founded_year=founded_year,
            funding_amount=funding_amount,
            funding_rounds=funding_rounds,
            investor_count=investor_count,
            team_size=team_size,
            revenue=revenue,
            profit_margin=profit_margin,
            customer_count=customer_count,
            growth_rate=growth_rate,
            founder_experience=founder_experience,
            market_size=market_size
        )

        # Now you have the predicted performance, you can do whatever you want with it
        return render(request, 'dashboard.html', {'prediction': performance})
    else:
        # Handle GET request (if any)
        return render(request, 'dashboard.html')
    
