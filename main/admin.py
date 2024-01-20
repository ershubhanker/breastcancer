from django.contrib import admin
from .models import Doctor,Patient, CreditRequest, EmailToken,Image
# Register your models here.

admin.site.register(Doctor)
admin.site.register(Patient)
admin.site.register(CreditRequest)
admin.site.register(EmailToken)
admin.site.register(Image)
