from django.urls import path
from . import views
from  django.contrib.auth import views as auth_views
from django.contrib.auth.forms import SetPasswordForm
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.views import PasswordResetConfirmView
from rest_framework.routers import DefaultRouter
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import GeneratePDFView

router = DefaultRouter()
router.register("api/upload-images", views.ImageprocessViewAPI, basename="upload-images")

urlpatterns = [
    # login registrations

    # path('verify/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='email_verification_sent'),
    # path('email-verification-sent/', views.email_verification_sent, name='email_verification_sent'),
    # path('verify/<uidb64>/<token>/', views.verify_email, name='verify_email'),
    path('verify/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='verify_email'),
    
    path('register/', views.doctor_register, name='doctor_register'),
    path('', views.doctor_login, name='doctor_login'),
    path('logout/', views.doctor_logout, name='doctor_logout'),
    # Generate pdf
    path('generate-pdf/', GeneratePDFView.as_view(), name='generate_pdf'),
    # home pages
    # path('home/', views.home, name='home'),
    # path('operator-home/', views.operator_home, name = "operator-home"),
    # admin home page
    path('home-admin/', views.adminhome, name='admin_home'),
    path('delete-image/<int:image_id>/', views.delete_image, name='delete_image'),
    # password recovery
    # path('reset_password/', auth_views.PasswordResetView.as_view(template_name = 'forget_password.html'), name='reset_password'),
    # path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name = 'email_sent.html'), name='password_reset_done'),
    # path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='new_password.html',form_class=SetPasswordForm), name='password_reset_confirm'),
    # path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name = 'password_change_success.html'), name='password_reset_complete'),

    path("reset_password/",views.Resetpasswordview.as_view() ,name="reset_password"),
    path("confirm/<str:token>/", views.confirmpassword.as_view(), name="confirm-password"),
    path('annotate-image/', views.annotate_image, name='annotate_image'),
    path('save_annotated_image/', views.save_annotated_image, name='save_annotated_image'),
    path('thermal-image/', views.thermal_image_view, name='thermal_image'),
    path('save_threshold/', views.save_threshold, name='save_threshold'),
    path('thermal-parameters/', views.thermal_parameters, name='thermal_parameters'),
    #doctor-dashboard
    path('index/', views.index, name = "index"),
    #doctor-profile
    path('doctor-profile/', views.DoctorProfile,  name = "DoctorProfile"),

    # doctor-services
    # path('services/', views.services, name = "services"),

    # doctor-image uploadings
    path('upload-images/', views.upload_image, name = "upload-image"),

    # doctor pending requests
    path('pending-request/', views.pendingRequest, name = "pendingRequest"),

    # doctor request successes
    path('success-request/', views.successRequest, name = "successRequest"),

    # doctor rejected requests
    path('rejected-request/', views.rejectedRequest, name = "rejectedRequest"),

    # doctor search reports
    path('search-reports/', views.searchReports, name = "searchReports"),

    # doctor request credits
    path('request-credit/', views.requestCredit, name = "requestCredit"),

    path('charts/', views.charts, name = "charts"),

    path('error/', views.error, name = "error"),

    path('mail_error/', views.mail_error, name = "mail_error"),

    path('general_error/', views.general_error, name = "general_error"),

    # get request credits list to admin
    path('credit_requset_list/', views.credit_requset_list, name = "credit_requset_list"),

    # path('api/send_report_data/', views.send_report_data, name='send_report_data'),

    # path('api/upload-images/', views.UploadImagesAPI.as_view(), name='upload_images_api'),
    path('api/token/', views.GetJwtToken.as_view(), name='token_obtain_pair'),
    # path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),


    ] + router.urls 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)