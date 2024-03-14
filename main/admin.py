from django.contrib import admin
from .models import UserForm, Image, Doctor, Assignment, PdfReport

class AssignmentInline(admin.TabularInline):
    model = Assignment
    fk_name = 'operator'

class ImageAdmin(admin.ModelAdmin):
    list_display = ['title', 'uploaded_at', 'is_annotated', 'assigned_doctor', 'uploader']
    search_fields = ['title']

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        elif hasattr(request.user, 'doctor_assignments'):
            assigned_doctors = request.user.doctor_assignments.values_list('doctor', flat=True)
            return qs.filter(assigned_doctor__in=assigned_doctors)
        elif hasattr(request.user, 'operator_assignments'):
            return qs.filter(uploader=request.user)
        return qs.none()

    def has_view_or_change_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        if obj is not None:
            if hasattr(request.user, 'doctor_assignments') and obj.assigned_doctor in request.user.doctor_assignments.all():
                return True
            if hasattr(request.user, 'operator_assignments') and obj.uploader == request.user:
                return True
            return False
        return super().has_view_or_change_permission(request, obj)
class DoctorAdmin(admin.ModelAdmin):
    inlines = [AssignmentInline]

admin.site.register(UserForm)
admin.site.register(PdfReport)
admin.site.register(Image, ImageAdmin)
admin.site.register(Doctor, DoctorAdmin)