from email import message
from django.core.mail import send_mail
import random
from . models import *

from django.conf  import settings


def Sendresetpasswordlink(receiver_email, token):

    try:
        subject = f'RESET EMAIL PASSWORD'
        message = f'Click below link to reset your password http://127.0.0.1:8000/confirm/{token}' 
        email_from = settings.EMAIL_HOST
        send_mail(subject, message, email_from, [receiver_email])
    except Exception as e:
        print(e)