from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from InvestMagnet.models import Message

# Create your views here.

def index(request):
    return render(request,'Home/index.html')



@login_required
def inbox(request):
    
    received_messages = Message.objects.filter(to_user=request.user)
    return render(request, 'Home/inbox.html', {'messages': received_messages})
