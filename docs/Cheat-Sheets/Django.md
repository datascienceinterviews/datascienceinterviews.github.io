
---
title: Django Cheat Sheet
description: A comprehensive reference guide for Django, covering commands, models, views, templates, forms, security, testing, deployment, caching, signals, and more.
---

# Django Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the Django web framework, covering essential commands, concepts, and code snippets for efficient Django development. It aims to be a one-stop reference for common tasks and best practices.

## Getting Started

### Installation

```bash
pip install django
```

Consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

### Create a Project

```bash
django-admin startproject myproject
cd myproject
```

### Create an App

```bash
python manage.py startapp myapp
```

### Run the Development Server

```bash
python manage.py runserver
```

## Models

### Define a Model (models.py)

```python
from django.db import models

class MyModel(models.Model):
    name = models.CharField(max_length=100, help_text="Enter the name")
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['order', '-created_at']  # Default ordering
        verbose_name = "My Model Entry"
        verbose_name_plural = "My Model Entries"
```

### Model Fields

*   `AutoField`: An auto-incrementing integer field (primary key by default).
*   `BigAutoField`: A 64-bit integer, similar to AutoField.
*   `CharField`: For short to medium-length strings. `max_length` is required.
*   `TextField`: For long strings of unlimited length. `blank=True` allows the field to be empty in forms, `null=True` allows the field to store NULL values in the database.
*   `IntegerField`: For integer values.
*   `PositiveIntegerField`: An integer field that must be positive.
*   `SmallIntegerField`, `BigIntegerField`: Smaller and larger integer fields.
*   `FloatField`: For floating-point numbers.
*   `DecimalField`: For fixed-precision decimal numbers. Requires `max_digits` and `decimal_places`.
*   `BooleanField`: For boolean values (True/False).
*   `NullBooleanField`: A BooleanField that also accepts NULL.
*   `DateField`: For dates (YYYY-MM-DD).
*   `DateTimeField`: For dates and times (YYYY-MM-DD HH:MM:SS). `auto_now_add=True` sets the field to the current date/time when the object is created. `auto_now=True` updates the field every time the object is saved.
*   `TimeField`: For times (HH:MM:SS).
*   `DurationField`: Stores periods of time – modeled in Python by timedelta.
*   `EmailField`: A CharField that validates if the input is an email address.
*   `URLField`: A CharField that validates URLs.
*   `FileField`: For file uploads. Requires `upload_to` to specify the storage directory.
*   `ImageField`: For image uploads. Requires Pillow library and `upload_to`.
*   `FilePathField`: A CharField whose choices are limited to the filenames in a certain directory on the filesystem.
*   `SlugField`: A CharField intended to store a "slug" – a short label containing only letters, numbers, underscores or hyphens.
*   `BinaryField`: For storing raw binary data.
*   `ForeignKey`: For creating one-to-many relationships with other models. Requires `on_delete` to specify what happens when the related object is deleted (e.g., `models.CASCADE`, `models.SET_NULL`).
*   `ManyToManyField`: For creating many-to-many relationships.
*   `OneToOneField`: For creating one-to-one relationships.
*   `GenericIPAddressField`: For storing IPv4 or IPv6 addresses.
*   `UUIDField`: For storing universally unique identifiers.

### Model Meta Options

*   `ordering`: Defines the default ordering of objects.
*   `verbose_name`: A human-readable name for the model.
*   `verbose_name_plural`: The plural form of the verbose name.
*   `abstract = True`: Makes the model an abstract base class.
*   `db_table`: Specifies the name of the database table.
*   `unique_together`: Defines a set of fields that, taken together, must be unique.
*   `index_together`: Defines a set of fields that should be indexed together.
*   `get_latest_by`: Specifies a field to use for retrieving the "latest" object.

### Querying the Database

```python
from .models import MyModel

# Get all objects
all_objects = MyModel.objects.all()

# Filter objects
filtered_objects = MyModel.objects.filter(name__contains='keyword', is_active=True)

# Get a single object by primary key
single_object = MyModel.objects.get(pk=1)

# Get a single object, handling DoesNotExist exception
from django.shortcuts import get_object_or_404
single_object = get_object_or_404(MyModel, pk=1)

# Create a new object
new_object = MyModel.objects.create(name='New Object', description='...')

# Update an existing object
obj = MyModel.objects.get(pk=1)
obj.name = 'Updated Name'
obj.save()

# Delete an object
obj = MyModel.objects.get(pk=1)
obj.delete()

# Complex lookups with Q objects
from django.db.models import Q
objects = MyModel.objects.filter(Q(name__startswith='A') | Q(description__icontains='data'))

# Ordering
ordered_objects = MyModel.objects.order_by('name', '-created_at')

# Limiting results
limited_objects = MyModel.objects.all()[:10]

# Chaining queries
chained_objects = MyModel.objects.filter(is_active=True).order_by('name')
```

### Raw SQL Queries

```python
from django.db import connection

def my_raw_query():
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM myapp_mymodel WHERE name = %s", ['My Name'])
        row = cursor.fetchone()
        return row
```

## Views

### Create a View (views.py)

```python
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from .models import MyModel

def my_view(request):
    data = MyModel.objects.all()  # Get all objects from MyModel
    context = {'data': data}
    return render(request, 'myapp/mytemplate.html', context)

def detail_view(request, pk):
    item = get_object_or_404(MyModel, pk=pk)
    return render(request, 'myapp/detail.html', {'item': item})

def json_response(request):
    data = {'message': 'Hello, world!'}
    return JsonResponse(data)

def http_response(request):
    return HttpResponse("<h1>Hello, world!</h1>", content_type="text/html")
```

### Class-Based Views

```python
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from .models import MyModel

class MyListView(ListView):
    model = MyModel
    template_name = 'myapp/mymodel_list.html'
    context_object_name = 'data'  # Renames the object_list in the template
    paginate_by = 10  # Enable pagination

class MyDetailView(DetailView):
    model = MyModel
    template_name = 'myapp/mymodel_detail.html'
    context_object_name = 'item'

class MyCreateView(CreateView):
    model = MyModel
    fields = ['name', 'description', 'is_active']  # Fields to include in the form
    template_name = 'myapp/mymodel_form.html'
    success_url = reverse_lazy('myapp:my_list_view')  # Redirect after successful creation

class MyUpdateView(UpdateView):
    model = MyModel
    fields = ['name', 'description', 'is_active']
    template_name = 'myapp/mymodel_form.html'
    success_url = reverse_lazy('myapp:my_list_view')

class MyDeleteView(DeleteView):
    model = MyModel
    template_name = 'myapp/mymodel_confirm_delete.html'
    success_url = reverse_lazy('myapp:my_list_view')
```

### Function-Based View Decorators

*   `@require_http_methods(["GET", "POST"])`: Only allows specified HTTP methods.
*   `@require_GET`, `@require_POST`: Shorthand for requiring GET or POST.
*   `@login_required`: Requires the user to be logged in.
*   `@permission_required('myapp.change_mymodel')`: Requires the user to have a specific permission.
*   `@staff_member_required`: Requires the user to be a staff member.
*   `@cache_page(60 * 15)`: Caches the view output for 15 minutes (requires cache configuration).

## URLs

### Define URL Patterns (urls.py)

Project `urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('myapp/', include('myapp.urls', namespace='myapp')),  # Include app's URLs with namespace
]
```

App `urls.py`:

```python
from django.urls import path
from . import views

app_name = 'myapp'  # App namespace

urlpatterns = [
    path('', views.my_view, name='my_view'),
    path('list/', views.MyListView.as_view(), name='my_list_view'),
    path('detail/<int:pk>/', views.MyDetailView.as_view(), name='my_detail_view'),
    path('create/', views.MyCreateView.as_view(), name='my_create_view'),
    path('update/<int:pk>/', views.MyUpdateView.as_view(), name='my_update_view'),
    path('delete/<int:pk>/', views.MyDeleteView.as_view(), name='my_delete_view'),
]
```

### URL Reversing

In templates:

```html
<a href="{% url 'myapp:my_detail_view' item.pk %}">{{ item.name }}</a>
```

In Python code:

```python
from django.urls import reverse

url = reverse('myapp:my_detail_view', kwargs={'pk': 1})
```

## Templates

### Create a Template (mytemplate.html)

```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My Template{% endblock %}</title>
    {% load static %}
    <link rel="stylesheet" href="{% static 'myapp/css/style.css' %}">
</head>
<body>
    <header>
        <h1>{% block header %}My Website{% endblock %}</h1>
    </header>

    <main>
        {% block content %}
            <h1>Data from MyModel:</h1>
            <ul>
                {% for item in data %}
                    <li><a href="{% url 'myapp:my_detail_view' item.pk %}">{{ item.name }}</a> - {{ item.description }}</li>
                {% empty %}
                    <li>No data available.</li>
                {% endfor %}
            </ul>
        {% endblock %}
    </main>

    <footer>
        <p>&copy; 2025 My Website</p>
    </footer>
</body>
</html>
```

### Template Inheritance

Create a base template (`base.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My Website{% endblock %}</title>
</head>
<body>
    <header>
        <h1>{% block header %}My Website{% endblock %}</h1>
    </header>

    <main>
        {% block content %}{% endblock %}
    </main>

    <footer>
        <p>&copy; 2025 My Website</p>
    </footer>
</body>
</html>
```

Extend the base template (`mytemplate.html`):

```html
{% extends 'base.html' %}

{% block title %}My Custom Title{% endblock %}

{% block content %}
    <h1>Data from MyModel:</h1>
    <ul>
        {% for item in data %}
            <li>{{ item.name }} - {{ item.description }}</li>
        {% endfor %}
    </ul>
{% endblock %}
```

### Template Tags and Filters

*   `{{ variable }}`: Outputs a variable.
*   `{% tag %}`: Template logic tag (e.g., `for`, `if`).
*   `{{ variable|filter }}`: Applies a filter to a variable.
*   `{% load static %}`: Loads the `static` template tag library for serving static files.
*   `{% url 'view_name' arg1 arg2 %}`: Reverses a URL pattern by its name.
*   `{% csrf_token %}`: Adds a CSRF token to a form.
*   `{% now "Y-m-d H:i" %}`: Displays the current date and time.
*   `{% include "template_name.html" %}`: Includes another template.
*   `{% extends "base.html" %}`: Extends a base template.
*   `{% block block_name %}{% endblock %}`: Defines a block for template inheritance.
*   `{% if condition %}{% endif %}`: Conditional logic.
*   `{% for item in items %}{% endfor %}`: Loop through a list.
*   `{% with total=items|length %}`: Assign a value to a variable within the template.

### Common Template Filters

*   `safe`: Marks a string as safe for HTML output.
*   `date:"FORMAT_STRING"`: Formats a date. See Django's documentation for format string options.
*   `time:"FORMAT_STRING"`: Formats a time.
*   `timesince`: Displays the time elapsed since a date.
*   `truncatechars:LENGTH`: Truncates a string to a certain length.
*   `truncatewords:NUM`: Truncates a string to a certain number of words.
*   `lower`, `upper`: Converts a string to lowercase or uppercase.
*   `title`: Converts a string to title case.
*   `capfirst`: Capitalizes the first character of a string.
*   `length`: Returns the length of a value.
*   `default:VALUE`: Provides a default value if a variable is False.
*   `filesizeformat`: Formats a number as a human-readable file size.
*   `stringformat:"E"`: Formats a number according to a string format specifier.
*   `linebreaks`: Replaces line breaks in plain text with appropriate HTML; a single newline becomes an HTML line break (`<br>`) and a new line surrounded by empty lines becomes a paragraph break (`<p>`).
*   `urlencode`: Encodes a string for use in a URL.
*   `json_script`: Safely outputs data as JSON for use in JavaScript.

## Forms

### Define a Form (forms.py)

```python
from django import forms
from .models import MyModel

class MyForm(forms.Form):
    name = forms.CharField(label="Your Name", max_length=100,
                           widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label="Your Email",
                            widget=forms.EmailInput(attrs={'class': 'form-control'}))
    message = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}),
                              label="Your Message")
    agree = forms.BooleanField(label="I agree to the terms", required=True)

    # Custom validation
    def clean_name(self):
        name = self.cleaned_data['name']
        if len(name) < 3:
            raise forms.ValidationError("Name must be at least 3 characters long.")
        return name

class MyModelForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = ['name', 'description', 'is_active']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        }
        labels = {
            'name': 'Model Name',
            'description': 'Model Description',
        }
        help_texts = {
            'name': 'Enter a descriptive name for the model.',
        }
        error_messages = {
            'name': {
                'required': 'Please enter a name.',
            },
        }
```

### Form Fields

*   `CharField`: For text input.
*   `IntegerField`: For integer input.
*   `FloatField`: For floating-point input.
*   `BooleanField`: For checkbox input.
*   `DateField`, `DateTimeField`: For date and time input.
*   `EmailField`: For email input.
*   `URLField`: For URL input.
*   `ChoiceField`: For select input. Requires `choices` argument.
*   `MultipleChoiceField`: For multiple select input.
*   `FileField`: For file upload.
*   `ImageField`: For image upload.
*   `ModelChoiceField`: For selecting a model instance from a queryset.
*   `ModelMultipleChoiceField`: For selecting multiple model instances.
*   `TypedChoiceField`: Like `ChoiceField`, but coerces values to a specific type.
*   `TypedMultipleChoiceField`: Like `MultipleChoiceField`, but coerces values to a specific type.
*   `RegexField`: A CharField that validates against a regular expression.

### Form Widgets

*   `TextInput`: Default text input.
*   `Textarea`: Multi-line text input.
*   `NumberInput`: For number input.
*   `EmailInput`: For email input.
*   `URLInput`: For URL input.
*   `PasswordInput`: For password input.
*   `HiddenInput`: A hidden input field.
*   `Select`: For single select dropdown.
*   `SelectMultiple`: For multiple select dropdown.
*   `RadioSelect`: For radio button selection.
*   `CheckboxInput`: For a single checkbox.
*   `CheckboxSelectMultiple`: For multiple checkboxes.
*   `FileInput`: For file uploads.
*   `ClearableFileInput`: A FileInput with a checkbox to clear the current file.
*   `DateInput`, `DateTimeInput`, `TimeInput`: For date, datetime, and time input, respectively.

### Render a Form in a Template

```html
<form method="post">
    {% csrf_token %}
    {% if form.errors %}
        <div class="alert alert-danger">
            Please correct the errors below.
        </div>
    {% endif %}
    {{ form.as_p }}  {# Renders the form as a series of <p> tags #}
    {# Or render fields individually: #}
    {# <div class="form-group">
        {{ form.name.label_tag }}
        {{ form.name }}
        {{ form.name.errors }}
    </div> #}
    <button type="submit">Submit</button>
</form>
```

### Process Form Data in a View

```python
from django.shortcuts import render, redirect
from .forms import MyForm, MyModelForm

def my_form_view(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']
            # Process the data (e.g., save to database, send email)
            return redirect('success_url')  # Redirect to a success page
        else:
            # Form is invalid, display errors
            return render(request, 'myapp/myform.html', {'form': form})
    else:
        form = MyForm()
    return render(request, 'myapp/myform.html', {'form': form})

def my_model_form_view(request):
    if request.method == 'POST':
        form = MyModelForm(request.POST, request.FILES) # Include request.FILES for file uploads
        if form.is_valid():
            instance = form.save()  # Save the model instance
            # Or, to process data before saving:
            # new_instance = form.save(commit=False)
            # new_instance.some_field = 'some_value'
            # new_instance.save()
            return redirect('my_list_view')
        else:
            return render(request, 'myapp/mymodel_form.html', {'form': form})
    else:
        form = MyModelForm()
    return render(request, 'myapp/mymodel_form.html', {'form': form})
```

## Admin Interface

### Register a Model (admin.py)

```python
from django.contrib import admin
from .models import MyModel

admin.site.register(MyModel)
```

### Customize Admin Interface

```python
from django.contrib import admin
from .models import MyModel

@admin.register(MyModel)
class MyModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'is_active')  # Columns to display in list view
    search_fields = ('name', 'description') # Enable search
    list_filter = ('is_active', 'created_at')          # Enable filtering
    ordering = ('name',)
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at' # Drill-down by date
    prepopulated_fields = {'slug': ('name',)} # Automatically populate slug field
    raw_id_fields = ('related_model',) # Use raw ID lookup for ForeignKey/ManyToManyField
    filter_horizontal = ('many_to_many_field',) # Use horizontal filter for ManyToManyField
    filter_vertical = ('another_many_to_many_field',) # Use vertical filter for ManyToManyField
    fieldsets = (
        (None, {
            'fields': ('name', 'description')
        }),
        ('Advanced options', {
            'classes': ('collapse',),
            'fields': ('is_active', 'order'),
        }),
    )
    actions = ['make_active', 'make_inactive']

    def make_active(self, request, queryset):
        queryset.update(is_active=True)
    make_active.short_description = "Mark selected entries as active"

    def make_inactive(self, request, queryset):
        queryset.update(is_active=False)
    make_inactive.short_description = "Mark selected entries as inactive"
```

### Inline Admin

```python
from django.contrib import admin
from .models import MyModel, RelatedModel

class RelatedModelAdminInline(admin.TabularInline):  # Or admin.StackedInline
    model = RelatedModel
    extra = 1  # Number of empty forms to display
    fk_name = 'mymodel' # Specify the ForeignKey field name in RelatedModel

@admin.register(MyModel)
class MyModelAdmin(admin.ModelAdmin):
    inlines = [RelatedModelAdminInline]
```

## Settings (settings.py)

### Key Settings

*   `DEBUG = True`: Enables debug mode (for development only!).
*   `SECRET_KEY`: A secret key for security. **Never share this!** Use environment variables.
*   `ALLOWED_HOSTS = ['*']`: List of allowed hostnames for the Django project. In production, set this to your domain name.
*   `INSTALLED_APPS`: List of installed Django apps.
*   `MIDDLEWARE`: List of enabled middleware.
*   `ROOT_URLCONF`: Specifies the root URL configuration module.
*   `DATABASES`: Database connection settings.
*   `STATIC_URL`: URL to serve static files.
*   `STATIC_ROOT`: The absolute path to the directory where `collectstatic` will collect static files for production.
*   `STATICFILES_DIRS`: List of directories where Django will look for static files.
*   `MEDIA_URL`: URL to serve media files (user-uploaded files).
*   `MEDIA_ROOT`: The absolute path to the directory where user-uploaded media files will be stored.
*   `TEMPLATES`: Template engine settings.
*   `LANGUAGE_CODE`: The default language code for the project.
*   `TIME_ZONE`: The timezone for the project.
*   `USE_I18N = True`: Enables internationalization.
*   `USE_L10N = True`: Enables localization.
*   `USE_TZ = True`: Enables timezone support.
*   `DEFAULT_AUTO_FIELD`: Default auto-field type for primary keys (Django 3.2+).
*   `SESSION_ENGINE`: Defines the session storage engine.
*   `CSRF_COOKIE_SECURE = True`: Ensures the CSRF cookie is only sent over HTTPS (production).
*   `SESSION_COOKIE_SECURE = True`: Ensures the session cookie is only sent over HTTPS (production).
*   `SECURE_SSL_REDIRECT = True`: Redirects all HTTP traffic to HTTPS (production).
*   `SECURE_HSTS_SECONDS = 31536000`: Enables HTTP Strict Transport Security (HSTS) (production).
*   `SECURE_HSTS_INCLUDE_SUBDOMAINS = True`: Includes subdomains in HSTS policy (production).
*   `SECURE_HSTS_PRELOAD = True`: Enables HSTS preloading (production).

### Database Configuration

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',  # Or 'django.db.backends.mysql', 'django.db.backends.sqlite3'
        'NAME': 'mydatabase',
        'USER': 'myuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Static Files Configuration

```python
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'myapp/static'),
]
```

### Middleware Configuration

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.cache.UpdateCacheMiddleware', # Add for caching
    'django.middleware.cache.FetchFromCacheMiddleware', # Add for caching
]
```

### Caching Configuration

```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379',
    }
}

CACHE_MIDDLEWARE_ALIAS = 'default'
CACHE_MIDDLEWARE_SECONDS = 600  # Cache for 10 minutes
CACHE_MIDDLEWARE_KEY_PREFIX = ''
```

### Email Configuration

```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your_email@gmail.com'
EMAIL_HOST_PASSWORD = 'your_password'
DEFAULT_FROM_EMAIL = 'your_email@gmail.com'
```

## Common Commands

*   `python manage.py runserver`: Starts the development server.
*   `python manage.py shell`: Opens a Python shell with Django environment loaded.
*   `python manage.py createsuperuser`: Creates an admin user.
*   `python manage.py makemigrations`: Creates new migrations based on model changes.
*   `python manage.py migrate`: Applies migrations to the database.
*   `python manage.py collectstatic`: Collects static files into `STATIC_ROOT`.
*   `python manage.py test`: Runs the project's tests.
*   `python manage.py dbshell`: Opens a shell for the database.
*   `python manage.py dumpdata`: Exports data from the database as JSON or XML.
*   `python manage.py loaddata`: Loads data from a JSON or XML fixture into the database.
*   `python manage.py check`: Performs system checks to identify potential problems.
*   `python manage.py showmigrations`: Shows the status of migrations.
*   `python manage.py inspectdb`: Generates models from an existing database.
*   `python manage.py flush`: Empties the database.
*   `python manage.py changepassword <username>`: Changes a user's password.

## Django REST Framework (DRF)

### Installation

```bash
pip install djangorestframework
```

Add `'rest_framework'` to `INSTALLED_APPS` in `settings.py`.

### Serializers (serializers.py)

```python
from rest_framework import serializers
from .models import MyModel

class MyModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyModel
        fields = '__all__'  # Or specify a tuple of field names
        # or exclude = ('field1', 'field2')
        read_only_fields = ('created_at', 'updated_at') # Make fields read-only

    # Custom field validation
    def validate_name(self, value):
        if len(value) < 5:
            raise serializers.ValidationError("Name must be at least 5 characters long.")
        return value

    # Object-level validation
    def validate(self, data):
        if data['name'] == data['description']:
            raise serializers.ValidationError("Name and description cannot be the same.")
        return data
```

### Views (views.py)

```python
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from .models import MyModel
from .serializers import MyModelSerializer

class MyModelList(generics.ListCreateAPIView):
    queryset = MyModel.objects.all()
    serializer_class = MyModelSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly] # Require authentication for write operations

    def perform_create(self, serializer):
        serializer.save() # Save the object

class MyModelDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = MyModel.objects.all()
    serializer_class = MyModelSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def delete(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT) # Return 204 No Content on successful deletion
```

### URLs (urls.py)

```python
from django.urls import path
from . import views

urlpatterns = [
    path('mymodel/', views.MyModelList.as_view(), name='mymodel-list'),
    path('mymodel/<int:pk>/', views.MyModelDetail.as_view(), name='mymodel-detail'),
]
```

### Authentication and Permissions

*   `permissions.AllowAny`: Allows access to anyone, authenticated or not.
*   `permissions.IsAuthenticated`: Only allows access to authenticated users.
*   `permissions.IsAdminUser`: Only allows access to admin users.
*   `permissions.IsAuthenticatedOrReadOnly`: Allows read access to anyone, but write access only to authenticated users.
*   Custom permissions: You can create custom permission classes to define more specific access control rules.

### APIView

For more control, use `APIView`:

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import MyModel
from .serializers import MyModelSerializer

class MyCustomAPIView(APIView):
    def get(self, request):
        data = MyModel.objects.all()
        serializer = MyModelSerializer(data, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = MyModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

## Security

### CSRF Protection

*   Use the `{% csrf_token %}` template tag in forms.
*   Ensure `django.middleware.csrf.CsrfViewMiddleware` is in `MIDDLEWARE`.

### SQL Injection

*   Use Django's ORM to avoid raw SQL queries whenever possible.
*   If you must use raw SQL, use parameterized queries to escape user input.

### XSS (Cross-Site Scripting)

*   Use the `safe` template filter with caution. Only use it on data you trust.
*   Sanitize user input before displaying it.

### Clickjacking

*   Ensure `django.middleware.clickjacking.XFrameOptionsMiddleware` is in `MIDDLEWARE`.
*   Set `X_FRAME_OPTIONS = 'DENY'` or `X_FRAME_OPTIONS = 'SAMEORIGIN'` in `settings.py`.

### Security Headers

Use `django-security` or similar package to set security headers.

### HTTPS

*   Configure your web server to use HTTPS.
*   Set `SECURE_SSL_REDIRECT = True` in `settings.py` to redirect HTTP requests to HTTPS.
*   Set `SECURE_HSTS_SECONDS` and `SECURE_HSTS_INCLUDE_SUBDOMAINS` for HTTP Strict Transport Security.
*   Set `SECURE_HSTS_PRELOAD = True` to enable HSTS preloading.

### Authentication

Use Django's built-in authentication
```python
import TestCase
from .models import MyModel

class MyModelTest(TestCase):
    def setUp(self):
        MyModel.objects.create(name='Test Object', description='Test Description')

    def test_model_content(self):
        obj = MyModel.objects.get(name='Test Object')
        self.assertEqual(obj.description, 'Test Description')
```

### Test Client

```python
from django.test import Client

class MyViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_my_view(self):
        response = self.client.get('/myapp/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'My Template')
```

## Deployment

### Production Settings

Create a `production.py` settings file:

```python
from .settings import *

DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
```

### Web Server (Gunicorn)

Install Gunicorn:

```bash
pip install gunicorn
```

Run Gunicorn:

```bash
gunicorn myproject.wsgi:application --bind 0.0.0.0:8000
```

### Process Manager (systemd)

Create a systemd service file (`/etc/systemd/system/myproject.service`):

```ini
[Unit]
Description=My Django Project
After=network.target

[Service]
User=myuser
Group=mygroup
WorkingDirectory=/path/to/myproject
ExecStart=/path/to/venv/bin/gunicorn myproject.wsgi:application --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable myproject
sudo systemctl start myproject
```

### Static Files

*   Run `python manage.py collectstatic` to collect static files.
*   Configure your web server (e.g., Nginx, Apache) to serve static files from `STATIC_ROOT`.

### Media Files

*   Configure your web server to serve media files from `MEDIA_ROOT`.
*   Consider using a cloud storage service (e.g., AWS S3) for media files.

### Database

*   Use a production-ready database (e.g., PostgreSQL, MySQL).
*   Configure the database connection settings in `settings.py`.
*   Back up your database regularly.

### Environment Variables

*   Use environment variables for sensitive settings (e.g., `SECRET_KEY`, database credentials).
*   Use a package like `python-dotenv` to manage environment variables.

## Caching

### Per-Site Cache

Add `django.middleware.cache.UpdateCacheMiddleware` and `django.middleware.cache.FetchFromCacheMiddleware` to `MIDDLEWARE` in `settings.py`.

```python
# settings.py
MIDDLEWARE = [
    'django.middleware.cache.UpdateCacheMiddleware',
    # ... other middleware ...
    'django.middleware.cache.FetchFromCacheMiddleware',
]
```

Configure the cache backend:

```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379',  # Example using Redis
    }
}

CACHE_MIDDLEWARE_ALIAS = 'default'
CACHE_MIDDLEWARE_SECONDS = 600  # Cache for 10 minutes
CACHE_MIDDLEWARE_KEY_PREFIX = ''
```

### Per-View Cache

Use the `@cache_page` decorator:

```python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Cache for 15 minutes
def my_view(request):
    # ...
    return render(request, 'myapp/mytemplate.html', context)
```

### Template Fragment Caching

```html
{% load cache %}

{% cache 600 "my_template_fragment" %}
    {# Expensive template code here #}
{% endcache %}
```

### Low-Level Cache API

```python
from django.core.cache import cache

# Set a value
cache.set('my_key', 'my_value', 600)  # Cache for 10 minutes

# Get a value
value = cache.get('my_key')

# Delete a value
cache.delete('my_key')
```

## Signals

### Define a Signal (signals.py)

```python
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import MyModel

@receiver(post_save, sender=MyModel)
def my_model_post_save(sender, instance, created, **kwargs):
    if created:
        # Perform actions when a new MyModel instance is created
        print(f"New MyModel instance created: {instance.name}")
    else:
        # Perform actions when a MyModel instance is updated
        print(f"MyModel instance updated: {instance.name}")
```

### Connect Signals (apps.py)

```python
from django.apps import AppConfig

class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        import myapp.signals  # Import the signals module
```

### Common Signals

*   `pre_save`, `post_save`: Sent before and after a model's `save()` method is called.
*   `pre_delete`, `post_delete`: Sent before and after a model instance is deleted.
*   `m2m_changed`: Sent when a ManyToManyField is changed.
*   `pre_migrate`, `post_migrate`: Sent before and after migrations are applied.

## Internationalization (i18n) and Localization (l10n)

### Enable i18n and l10n

Set `USE_I18N = True` and `USE_L10N = True` in `settings.py`.

### Set the Language Code

Set `LANGUAGE_CODE = 'en-us'` in `settings.py`.

### Translate Strings

In Python code:

```python
from django.utils.translation import gettext as _

def my_view(request):
    message = _("Hello, world!")
    return render(request, 'myapp/mytemplate.html', {'message': message})
```

In templates:

```html
{% load i18n %}
<h1>{% trans "Hello, world!" %}</h1>
```

### Mark Strings for Translation

Use `makemessages` command:

```bash
python manage.py makemessages -l de  # Create a translation file for German
```

### Translate Strings with Context

```python
from django.utils.translation import pgettext as _

message = _("context", "Hello, world!")
```

### Pluralization

```python
from django.utils.translation import ngettext

def my_view(request, count):
    message = ngettext(
        'There is %(count)d object',
        'There are %(count)d objects',
        count
    ) % {'count': count}
    return render(request, 'myapp/mytemplate.html', {'message': message})
```

### Switch Language

```html
<form action="{% url 'set_language' %}" method="post">
    {% csrf_token %}
    <input name="language" type="hidden" value="de">
    <button type="submit">Switch to German</button>
</form>
```

## Custom Management Commands

### Create a Command (management/commands/mycommand.py)

```python
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'My custom command'

    def add_arguments(self, parser):
        parser.add_argument('argument', nargs='?', type=str, help='An argument for the command')

    def handle(self, *args, **options):
        argument = options['argument']
        self.stdout.write(self.style.SUCCESS(f'Successfully executed mycommand with argument: {argument}'))
```

### Run the Command

```bash
python manage.py mycommand "My Argument"
```

## Middleware

### Create a Middleware (middleware.py)

```python
class MyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Code to be executed for each request before the view (pre-processing)
        print("Before view")

        response = self.get_response(request)

        # Code to be executed for each request after the view (post-processing)
        print("After view")

        return response
```

### Activate Middleware

Add the middleware to `MIDDLEWARE` in `settings.py`:

```python
MIDDLEWARE = [
    # ...
    'myapp.middleware.MyMiddleware',
]
```

## File Handling

### Uploading Files

In `forms.py`:

```python
class UploadFileForm(forms.Form):
    file = forms.FileField()
```

In `views.py`:

```python
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            # Process the uploaded file (e.g., save to MEDIA_ROOT)
            with open(os.path.join(settings.MEDIA_ROOT, uploaded_file.name), 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            return HttpResponse("File uploaded successfully")
    else:
        form = UploadFileForm()
    return render(request, 'myapp/upload.html', {'form': form})
```

In `templates/myapp/upload.html`:

```html
<form method="post" enctype="multipart/form-data">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Upload</button>
</form>
```

### Serving Files

Configure `MEDIA_URL` and `MEDIA_ROOT` in `settings.py`.

In `urls.py`:

```python
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # ... your other URL patterns ...
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

## Logging

### Configure Logging (settings.py)

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'debug.log'),
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    },
}
```

### Use Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_view(request):
    logger.info("My view was accessed")
    try:
        # ... some code that might raise an exception ...
    except Exception as e:
        logger.exception("An error occurred")
```

## Django Channels (Asynchronous)

### Installation

```bash
pip install channels
```

### Configure Channels (settings.py)

```python
ASGI_APPLICATION = 'myproject.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

### Create a Consumer (consumers.py)

```python
from channels.generic.websocket import WebsocketConsumer
import json

class MyConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        self.send(text_data=json.dumps({
            'message': message
        }))
```

### Configure Routing (routing.py)

```python
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/myapp/$', consumers.MyConsumer.as_asgi()),
]
```

### Update ASGI Application (asgi.py)

```python
import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import myapp.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            myapp.routing.websocket_urlpatterns
        )
    ),
})
```

## Django Allauth (Authentication)

### Installation

```bash
pip install django-allauth
```

### Configuration (settings.py)

```python
INSTALLED_APPS = [
    # ...
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    # ... include providers you want to use ...
    # 'allauth.socialaccount.providers.google',
]

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

SITE_ID = 1

LOGIN_REDIRECT_URL = '/'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_USERNAME_REQUIRED = False
ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'
```

### URLs (urls.py)

```python
from django.urls import include, path

urlpatterns = [
    path('accounts/', include('allauth.urls')),
]
```

### Templates

Use Allauth's template tags and forms for registration, login, etc.

## Django Debug Toolbar

### Installation

```bash
pip install django-debug-toolbar
```

### Configuration (settings.py)

```python
INSTALLED_APPS = [
    # ...
    'debug_toolbar',
]

MIDDLEWARE = [
    # ...
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

INTERNAL_IPS = [
    '127.0.0.1',
]
```

### URLs (urls.py)

```python
from django.urls import include, path

urlpatterns = [
    # ...
]

if settings.DEBUG:
    import debug_toolbar
    urlpatterns += [
        path('__debug__/', include(debug_toolbar.urls)),
    ]
```

## Tips and Best Practices

*   Use virtual environments to isolate project dependencies.
*   Keep `SECRET_KEY` secure and out of your codebase. Use environment variables.
*   Use meaningful names for models, views, and URLs.
*   Follow the DRY (Don't Repeat Yourself) principle.
*   Write unit tests to ensure code quality.
*   Use Django's built-in security features (e.g., CSRF protection).
*   Configure static file serving correctly in production.
*   Use a production-ready web server (e.g., Gunicorn, uWSGI) and a process manager (e.g., Supervisor, systemd) for deployment.
*   Use a linter (like `flake8`) and formatter (like `black`) to ensure consistent code style.
*   Use a well-defined project structure.
*   Keep your code modular and reusable.
*   Document your code.
*   Use a version control system (e.g., Git).
*   Follow Django's coding style guidelines.
*   Use Django's built-in caching mechanisms to improve performance.
*   Monitor your application for errors and performance issues.
*   Use a CDN (Content Delivery Network) for static files.
*   Optimize database queries.
*   Use asynchronous tasks for long-running operations (e.g., sending emails).
*   Implement proper logging and error handling.
*   Regularly update Django and its dependencies.
*   Use a security scanner to identify potential vulnerabilities.
*   Follow security best practices.