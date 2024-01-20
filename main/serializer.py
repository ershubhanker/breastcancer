from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
from enum import Enum

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.core.exceptions import ValidationError
from datetime import timedelta


class Dtypes(Enum):
    Type1 = "Type1"
    Type2 = "Type2"
   

    @classmethod
    def choices(cls):
        # print(tuple((i.name, i.value) for i in cls))
        return tuple((i.name, i.value) for i in cls)
    

class UploadImgeSerializer(serializers.Serializer):
    name = serializers.CharField()
    # email = serializers.EmailField()
    # diabetes_status  = serializers.BooleanField(default=False)
    # diabetes_type  = serializers.ChoiceField( choices=Dtypes.choices())

    # left_eye = serializers.ImageField(required=False, allow_null=True)
    # right_eye = serializers.ImageField(required=False, allow_null=True)
    # check_token= serializers.CharField()
 

class LoginSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        try:
            token = super().get_token(user)

            token["username"] = user.username
            token["is_superuser"] = user.is_superuser     #send the extra fields of User inside the token 

            return token
        except Exception as e:
            raise ValidationError("Something went wrong") 
        
    class Meta:
        # Set the token expiration time to 1 minutes (60 seconds)
        access_token_lifetime = timedelta(minutes=1)


