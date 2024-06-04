from django.shortcuts import render
from django.http import HttpResponse
from .startup_prediction import process_and_train_model
from .churn_prediction import train_and_predict_hr_model
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

        # Data for the graph
        categories = ['Founded Year', 'Funding Amount', 'Funding Rounds', 'Investor Count',
                      'Team Size',  'Profit Margin', 'Customer Count', 'Growth Rate',
                      'Founder Experience']
        values = [founded_year, funding_amount, funding_rounds, investor_count, team_size,
                  profit_margin, customer_count, growth_rate, founder_experience]

        # Pass the data to the template
        context = {
            'categories': categories,
            'values': values,
        }

        # Now you have the predicted performance, you can do whatever you want with it
        return render(request, 'dashboard.html', {'prediction': performance ,'categories': categories,
            'values': values} )
    else:
        # Handle GET request (if any)
        return render(request, 'dashboard.html')
    


def churn(request):
    return render(request, 'churn_prediction.html')


def predict_employee_status(request):
    if request.method == 'POST':
        # Extract data from POST request
        satisfaction_level = request.POST.get('satisfaction_level')
        last_evaluation = request.POST.get('last_evaluation')
        number_project = request.POST.get('number_project')
        average_montly_hours = request.POST.get('average_montly_hours')
        time_spend_company = request.POST.get('time_spend_company')
        work_accident = request.POST.get('work_accident')
        promotion_last_5years = request.POST.get('promotion_last_5years')
        salary = request.POST.get('salary')

        result = train_and_predict_hr_model(satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, promotion_last_5years, salary)

        categories = ['Satisfaction Level', 'Last Evaluation', 'Number of Projects', 'Average Monthly Hours',
              'Time Spent in Company', 'Work Accident', 'Promotion in Last 5 Years', 'Salary']
        values = [satisfaction_level, last_evaluation, number_project, average_montly_hours,
          time_spend_company, work_accident, promotion_last_5years, salary]

        return render(request,'churn_dashboard.html', {'prediction': result ,'categories': categories,
            'values': values})

    return render(request)