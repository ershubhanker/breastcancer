from django.contrib import admin
from .models import UserForm, Image, Doctor, Assignment

class AssignmentInline(admin.TabularInline):
    model = Assignment
    fk_name = 'operator'

class ImageAdmin(admin.ModelAdmin):
    list_display = ['title', 'uploaded_at', 'is_annotated', 'assigned_doctor']
    search_fields = ['title']

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        elif hasattr(request.user, 'doctor'):
            # If the user is a doctor, show only images assigned to them
            return qs.filter(assigned_doctor=request.user.doctor)
        else:
            # Hide all images for non-doctors and non-superusers
            return qs.none()

    def has_view_or_change_permission(self, request, obj=None):
        # Allow viewing and changing only if the user is the assigned doctor or a superuser
        if obj is not None and hasattr(request.user, 'doctor'):
            return obj.assigned_doctor == request.user.doctor
        return super().has_view_or_change_permission(request, obj)

class DoctorAdmin(admin.ModelAdmin):
    inlines = [AssignmentInline]

admin.site.register(UserForm)
admin.site.register(Image, ImageAdmin)
admin.site.register(Doctor, DoctorAdmin)