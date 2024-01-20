# This function allows the user to login with either username and password or phone number and password



from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.db.models import Q

User = get_user_model()

class UsernameOrMobileModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            user = User.objects.get(Q(username=username) | Q(mobile=username))
        except User.DoesNotExist:
            return None

        if user.check_password(password):
            return user
        return None