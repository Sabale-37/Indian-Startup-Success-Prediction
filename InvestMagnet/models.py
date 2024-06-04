from django.db import models
from django.contrib.auth.models import User

class Company(models.Model):
    start_year = models.IntegerField()
    description = models.TextField()
    username = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='company_images/')
    revenue = models.DecimalField(max_digits=15, decimal_places=2)
    founder = models.CharField(max_length=255)
    investment_round = models.CharField(max_length=255)

    def __str__(self):
        return self.founder
