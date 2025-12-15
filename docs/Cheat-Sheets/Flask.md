
---
title: Flask Cheat Sheet
description: A comprehensive reference guide for Flask, covering setup, routing, templates, forms, databases, extensions, testing, deployment, and more.
---

# Flask Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the Flask micro web framework, covering essential commands, concepts, and code snippets for efficient Flask development. It aims to be a one-stop reference for common tasks and best practices.

## Getting Started

### Installation

```bash
pip install flask
```

Consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

### Flask Application Lifecycle

```
    ┌──────────────────┐
    │  Create App      │
    │  Flask(__name__) │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Configure App   │
    │  app.config[]    │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Register Routes │
    │  @app.route()    │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Initialize      │
    │  Extensions      │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Run Application │
    │  app.run()       │
    └──────────────────┘
```

### Basic App Structure

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

### Running the App

```bash
# Direct execution
python app.py

# Using Flask CLI
export FLASK_APP=app.py
flask run

# Run on specific host and port
flask run --host=0.0.0.0 --port=8000
```

## Routing

### Request Flow

```
    ┌──────────────┐
    │ HTTP Request │
    └──────┬───────┘
           │
           ↓
    ┌──────────────────┐
    │  URL Matching    │
    │  @app.route()    │
    └──────┬───────────┘
           │
           ↓
    ┌──────────────────┐
    │  View Function   │
    │  Execute Logic   │
    └──────┬───────────┘
           │
           ↓
    ┌──────────────────┐
    │ Return Response  │
    │  HTML/JSON/etc   │
    └──────────────────┘
```

### Basic Route

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/about')
def about():
    return 'About Page'

if __name__ == '__main__':
    app.run(debug=True)
```

### Dynamic Routes

```python
from flask import Flask

app = Flask(__name__)

# String parameter (default)
@app.route('/user/<username>')
def show_user_profile(username):
    return f'User: {username}'

# Integer parameter
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post {post_id}'

# Float parameter
@app.route('/price/<float:amount>')
def show_price(amount):
    return f'Price: ${amount:.2f}'

# Path parameter (accepts slashes)
@app.route('/path/<path:subpath>')
def show_path(subpath):
    return f'Path: {subpath}'

# UUID parameter
@app.route('/uuid/<uuid:id>')
def show_uuid(id):
    return f'UUID: {id}'

if __name__ == '__main__':
    app.run(debug=True)
```

### HTTP Methods

```
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   GET    │     │   POST   │     │  DELETE  │
    │ Retrieve │     │  Create  │     │  Remove  │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                 │
         └────────────────┴─────────────────┘
                          ↓
                 ┌────────────────┐
                 │  Flask Route   │
                 │  View Function │
                 └────────────────┘
```

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Multiple methods on one route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        return f"Logging in: {username}"
    return "Show login form"

# RESTful API example
@app.route('/api/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
def user_api(user_id):
    if request.method == 'GET':
        return jsonify({'user_id': user_id, 'action': 'fetch'})
    elif request.method == 'PUT':
        data = request.get_json()
        return jsonify({'user_id': user_id, 'action': 'update', 'data': data})
    elif request.method == 'DELETE':
        return jsonify({'user_id': user_id, 'action': 'delete'})

# Separate routes for different methods
@app.route('/resource', methods=['GET'])
def get_resource():
    return "GET resource"

@app.route('/resource', methods=['POST'])
def create_resource():
    return "POST resource"

if __name__ == '__main__':
    app.run(debug=True)
```

### URL Building

```python
from flask import Flask, url_for, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index'

@app.route('/login')
def login():
    return 'Login'

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'

@app.route('/admin')
def admin():
    # Redirect to login
    return redirect(url_for('login'))

@app.route('/user-redirect/<username>')
def user_redirect(username):
    # Redirect to user profile
    return redirect(url_for('profile', username=username))

# Generate URLs programmatically
with app.test_request_context():
    print(url_for('index'))                              # Output: /
    print(url_for('login'))                              # Output: /login
    print(url_for('login', next='/'))                    # Output: /login?next=%2F
    print(url_for('profile', username='John Doe'))       # Output: /user/John%20Doe
    print(url_for('index', _external=True))              # Output: http://localhost/
    print(url_for('static', filename='style.css'))       # Output: /static/style.css

if __name__ == '__main__':
    app.run(debug=True)
```

## Templates

### Template Rendering Flow

```
    ┌────────────────┐
    │  View Function │
    └───────┬────────┘
            │
            ↓
    ┌────────────────┐
    │ render_template│
    └───────┬────────┘
            │
            ↓
    ┌────────────────┐
    │  Jinja2 Engine │
    │  Process Template
    └───────┬────────┘
            │
            ↓
    ┌────────────────┐
    │  HTML Response │
    └────────────────┘
```

### Basic Template Rendering

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/users')
def users():
    users_list = [
        {'id': 1, 'name': 'Alice', 'role': 'Admin'},
        {'id': 2, 'name': 'Bob', 'role': 'User'},
        {'id': 3, 'name': 'Charlie', 'role': 'User'}
    ]
    return render_template('users.html', users=users_list)

@app.route('/dashboard')
def dashboard():
    context = {
        'title': 'Dashboard',
        'user': {'name': 'John Doe', 'email': 'john@example.com'},
        'stats': {'views': 1500, 'posts': 42, 'comments': 128}
    }
    return render_template('dashboard.html', **context)

if __name__ == '__main__':
    app.run(debug=True)
```

### Template (templates/hello.html)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Hello</title>
</head>
<body>
    {% if name %}
        <h1>Hello {{ name }}!</h1>
    {% else %}
        <h1>Hello, World!</h1>
    {% endif %}
</body>
</html>
```

### Template Inheritance

Base template (`templates/base.html`):

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

Child template (`templates/hello.html`):

```html
{% extends "base.html" %}

{% block title %}Hello{% endblock %}

{% block content %}
    {% if name %}
        <h1>Hello {{ name }}!</h1>
    {% else %}
        <h1>Hello, World!</h1>
    {% endif %}
{% endblock %}
```

### Jinja2 Template Engine

```python
# Common Jinja2 Syntax:
# {{ variable }}                    - Output variable
# {% tag %}                         - Logic tag (if, for, etc.)
# {{ variable|filter }}             - Apply filter
# {# comment #}                     - Comment (not rendered)
# {% extends "base.html" %}         - Template inheritance
# {% block name %}{% endblock %}    - Define block
# {% include "partial.html" %}      - Include template
# {{ url_for('view_name') }}        - Generate URL
```

Example template (`templates/demo.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title|default('Default Title') }}</title>
</head>
<body>
    {# This is a comment #}
    
    {# Variables #}
    <h1>Hello {{ name|capitalize }}!</h1>
    
    {# Conditionals #}
    {% if user.is_admin %}
        <p>Welcome, Admin!</p>
    {% elif user.is_authenticated %}
        <p>Welcome, {{ user.name }}!</p>
    {% else %}
        <p>Please log in.</p>
    {% endif %}
    
    {# Loops #}
    <ul>
    {% for item in items %}
        <li>{{ loop.index }}. {{ item.name }} - ${{ item.price|round(2) }}</li>
    {% else %}
        <li>No items available</li>
    {% endfor %}
    </ul>
    
    {# URL generation #}
    <a href="{{ url_for('index') }}">Home</a>
    <a href="{{ url_for('profile', username='john') }}">Profile</a>
</body>
</html>
```

### Common Template Filters

```python
# String filters
{{ 'hello world'|capitalize }}        # 'Hello world'
{{ 'HELLO'|lower }}                   # 'hello'
{{ 'hello'|upper }}                   # 'HELLO'
{{ 'hello world'|title }}             # 'Hello World'
{{ '  text  '|trim }}                 # 'text'
{{ 'Hello <b>World</b>'|striptags }}  # 'Hello World'

# Numeric filters
{{ 42.5678|round }}                   # 43.0
{{ 42.5678|round(2) }}                # 42.57
{{ 42|abs }}                          # 42
{{ -42|abs }}                         # 42

# List filters
{{ [1, 2, 3]|length }}                # 3
{{ [3, 1, 2]|sort }}                  # [1, 2, 3]
{{ [1, 2, 3]|reverse }}               # [3, 2, 1]
{{ [1, 2, 3]|first }}                 # 1
{{ [1, 2, 3]|last }}                  # 3
{{ ['a', 'b', 'c']|join(', ') }}      # 'a, b, c'

# Conditional filters
{{ variable|default('N/A') }}         # Use default if undefined
{{ html_content|safe }}               # Mark as safe HTML
{{ none_value|default('Empty', true) }}  # Use default if falsy

# Date/Time filters (requires datetime object)
{{ date_obj|strftime('%Y-%m-%d') }}   # Format datetime

# Custom example
{{ 'hello {0}'|format('world') }}     # 'hello world'
{{ 'old text'|replace('old', 'new') }}  # 'new text'
```

## Forms

### Form Handling Flow

```
    ┌──────────────┐
    │  GET Request │
    │  Show Form   │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ User Fills   │
    │ Form & Submit│
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ POST Request │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │  Validation  │
    └──────┬───────┘
           │
      ┌────┴────┐
      │         │
   (Pass)    (Fail)
      │         │
      ↓         ↓
┌──────────┐ ┌──────────┐
│ Process  │ │  Show    │
│ & Redirect│ │  Errors  │
└──────────┘ └──────────┘
```

### Basic Form Handling

```python
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Process form data
        print(f"Name: {name}, Email: {email}, Message: {message}")
        
        return redirect(url_for('success'))
    
    return render_template('contact.html')

@app.route('/success')
def success():
    return "Form submitted successfully!"

if __name__ == '__main__':
    app.run(debug=True)
```

HTML template (`templates/contact.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>Contact Form</title>
</head>
<body>
    <h1>Contact Us</h1>
    <form method="post">
        <label for="name">Name:</label><br>
        <input type="text" id="name" name="name" required><br><br>
        
        <label for="email">Email:</label><br>
        <input type="email" id="email" name="email" required><br><br>
        
        <label for="message">Message:</label><br>
        <textarea id="message" name="message" rows="5" cols="40" required></textarea><br><br>
        
        <input type="submit" value="Submit">
    </form>
</body>
</html>
```

### Using Flask-WTF

Installation:

```bash
pip install flask-wtf
```

Configuration:

```python
import os
from flask import Flask
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length, EqualTo

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
```

Flask-WTF Flow:

```
    ┌──────────────┐
    │ Define Form  │
    │ Class (WTF)  │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ Instantiate  │
    │ in View      │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ Render in    │
    │ Template     │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │ Validate on  │
    │ Submit       │
    └──────┬───────┘
           │
      ┌────┴────┐
      │         │
   (Valid)  (Invalid)
      │         │
      ↓         ↓
┌──────────┐ ┌──────────┐
│ Process  │ │  Show    │
│ Data     │ │  Errors  │
└──────────┘ └──────────┘
```

### Define a Form (forms.py)

```python
from flask_wtf import FlaskForm
from wtforms import (
    StringField, 
    PasswordField, 
    SubmitField, 
    TextAreaField, 
    BooleanField,
    SelectField,
    RadioField,
    DateField
)
from wtforms.validators import (
    DataRequired, 
    Length, 
    Email, 
    EqualTo, 
    ValidationError,
    Regexp
)

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[
        DataRequired(message='Name is required'),
        Length(min=2, max=50, message='Name must be between 2 and 50 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email is required'),
        Email(message='Invalid email address')
    ])
    message = TextAreaField('Message', validators=[
        DataRequired(message='Message is required'),
        Length(min=10, max=500, message='Message must be between 10 and 500 characters')
    ])
    submit = SubmitField('Send Message')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=4, max=20),
        Regexp('^[A-Za-z0-9_]+$', message='Username must contain only letters, numbers, and underscores')
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    agree = BooleanField('I agree to the Terms and Conditions', validators=[
        DataRequired(message='You must agree to the terms')
    ])
    submit = SubmitField('Sign Up')

class ProfileForm(FlaskForm):
    bio = TextAreaField('Bio', validators=[Length(max=500)])
    country = SelectField('Country', choices=[
        ('', 'Select Country'),
        ('us', 'United States'),
        ('uk', 'United Kingdom'),
        ('ca', 'Canada')
    ])
    gender = RadioField('Gender', choices=[
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')
    ])
    birthdate = DateField('Birthdate', format='%Y-%m-%d')
    submit = SubmitField('Update Profile')
```

### Render a Form in a Template

```html
<form method="post">
    {{ form.csrf_token }}
    <div class="form-group">
        {{ form.name.label }}<br>
        {{ form.name(class="form-control") }}
        {% if form.name.errors %}
            <ul class="errors">
                {% for error in form.name.errors %}
                    <li>{{ error }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
    <div class="form-group">
        {{ form.email.label }}<br>
        {{ form.email(class="form-control") }}
        {% if form.email.errors %}
            <ul class="errors">
                {% for error in form.email.errors %}
                    <li>{{ error }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
    <div class="form-group">
        {{ form.message.label }}<br>
        {{ form.message(class="form-control") }}
        {% if form.message.errors %}
            <ul class="errors">
                {% for error in form.message.errors %}
                    <li>{{ error }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
    <div class="form-group">
        {{ form.agree.label }}
        {{ form.agree }}
        {% if form.agree.errors %}
            <ul class="errors">
                {% for error in form.agree.errors %}
                    <li>{{ error }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
    {{ form.submit(class="btn btn-primary") }}
</form>
```

### Process Form Data in a View

```python
from flask import Flask, render_template, request, redirect, url_for
from forms import MyForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/form', methods=['GET', 'POST'])
def my_form_view():
    form = MyForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        message = form.message.data
        # Process the data (e.g., save to database, send email)
        return redirect(url_for('success'))
    return render_template('myform.html', form=form)

@app.route('/success')
def success():
    return "Form submitted successfully!"

if __name__ == '__main__':
    app.run(debug=True)
```

## Databases

### Database Integration Flow

```
    ┌──────────────────┐
    │  Configure DB    │
    │  SQLAlchemy      │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Define Models   │
    │  db.Model        │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  Create Tables   │
    │  db.create_all() │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────┐
    │  CRUD Operations │
    │  Query, Add, etc │
    └──────────────────┘
```

### Using Flask-SQLAlchemy

Installation:

```bash
pip install flask-sqlalchemy
```

Configuration:

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False  # Set to True to log SQL queries

db = SQLAlchemy(app)
```

Database URI examples:

```python
# SQLite (local file)
'sqlite:///database.db'
'sqlite:////absolute/path/to/database.db'

# PostgreSQL
'postgresql://username:password@localhost:5432/database_name'

# MySQL
'mysql://username:password@localhost:3306/database_name'

# MySQL with PyMySQL
'mysql+pymysql://username:password@localhost:3306/database_name'
```

### Define Models

```python
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    posts = db.relationship('Post', backref='author', lazy='dynamic', cascade='all, delete-orphan')
    profile = db.relationship('Profile', backref='user', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f"<User {self.username}>"

class Post(db.Model):
    __tablename__ = 'posts'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(200), unique=True, nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    published = db.Column(db.Boolean, default=False)
    views = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Many-to-many relationship
    tags = db.relationship('Tag', secondary='post_tags', backref=db.backref('posts', lazy='dynamic'))

    def __repr__(self):
        return f"<Post {self.title}>"

class Profile(db.Model):
    __tablename__ = 'profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    bio = db.Column(db.Text)
    avatar = db.Column(db.String(200), default='default.jpg')
    website = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f"<Profile for User {self.user_id}>"

class Tag(db.Model):
    __tablename__ = 'tags'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

    def __repr__(self):
        return f"<Tag {self.name}>"

# Association table for many-to-many relationship
post_tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('posts.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tags.id'), primary_key=True),
    db.Column('created_at', db.DateTime, default=datetime.utcnow)
)
```

### Create and Manage Tables

```python
from yourapp import app, db

# Create all tables
with app.app_context():
    db.create_all()

# Drop all tables (careful!)
with app.app_context():
    db.drop_all()

# Recreate all tables (drop and create)
with app.app_context():
    db.drop_all()
    db.create_all()

# Add sample data
with app.app_context():
    # Create user
    user = User(username='john_doe', email='john@example.com', password_hash='hashed_password')
    db.session.add(user)
    db.session.commit()
    
    # Create post for user
    post = Post(title='First Post', slug='first-post', content='Hello World!', user_id=user.id)
    db.session.add(post)
    db.session.commit()
    
    # Create multiple objects
    users = [
        User(username='alice', email='alice@example.com', password_hash='hash1'),
        User(username='bob', email='bob@example.com', password_hash='hash2')
    ]
    db.session.add_all(users)
    db.session.commit()
```

### Querying the Database

```python
from yourapp import db, User, Post, Tag
from sqlalchemy import and_, or_, not_

# Basic queries
all_users = User.query.all()                              # Get all users
first_user = User.query.first()                           # Get first user
user = User.query.get(1)                                  # Get by primary key
user = User.query.get_or_404(1)                           # Get or return 404

# Filtering
user = User.query.filter_by(username='john_doe').first()  # Simple filter
users = User.query.filter(User.username == 'john').all()  # Complex filter
active_users = User.query.filter_by(is_active=True).all()

# Multiple conditions
users = User.query.filter(
    and_(
        User.is_active == True,
        User.created_at > datetime(2024, 1, 1)
    )
).all()

users = User.query.filter(
    or_(
        User.username == 'john',
        User.username == 'alice'
    )
).all()

# Ordering
users = User.query.order_by(User.created_at.desc()).all()
users = User.query.order_by(User.username.asc()).all()

# Limiting
users = User.query.limit(10).all()
users = User.query.offset(5).limit(10).all()

# Pagination
page = 1
per_page = 20
pagination = User.query.paginate(page=page, per_page=per_page, error_out=False)
users = pagination.items
total = pagination.total

# Counting
user_count = User.query.count()
active_count = User.query.filter_by(is_active=True).count()

# Like queries (pattern matching)
users = User.query.filter(User.username.like('%john%')).all()
users = User.query.filter(User.email.ilike('%@GMAIL.COM')).all()  # Case-insensitive

# In queries
usernames = ['alice', 'bob', 'charlie']
users = User.query.filter(User.username.in_(usernames)).all()

# Relationships
user = User.query.get(1)
user_posts = user.posts.all()                             # Get user's posts
user_posts = user.posts.filter_by(published=True).all()   # Filter related posts

post = Post.query.get(1)
author = post.author                                      # Get post's author

# Joins
results = db.session.query(User, Post).join(Post).all()
results = User.query.join(Post).filter(Post.published == True).all()

# Eager loading (avoid N+1 queries)
from sqlalchemy.orm import joinedload
users = User.query.options(joinedload(User.posts)).all()

# CREATE
new_user = User(username='jane_doe', email='jane@example.com', password_hash='hash')
db.session.add(new_user)
db.session.commit()

# UPDATE
user = User.query.get(1)
user.email = 'newemail@example.com'
db.session.commit()

# Or update multiple fields
User.query.filter_by(id=1).update({'email': 'updated@example.com', 'is_active': True})
db.session.commit()

# DELETE
user = User.query.get(1)
db.session.delete(user)
db.session.commit()

# Or delete with filter
User.query.filter_by(is_active=False).delete()
db.session.commit()

# Bulk operations
users = User.query.filter(User.is_active == False).all()
for user in users:
    db.session.delete(user)
db.session.commit()

# Rollback on error
try:
    user = User(username='test', email='test@example.com', password_hash='hash')
    db.session.add(user)
    db.session.commit()
except Exception as e:
    db.session.rollback()
    print(f"Error: {e}")
```

## Static Files

### Static File Structure

```
your_app/
    ├── app.py
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/
        │   └── style.css
        ├── js/
        │   └── script.js
        └── images/
            └── logo.png
```

### Configure Static Files

```python
from flask import Flask, url_for

app = Flask(__name__, 
    static_folder='static',      # Default: 'static'
    static_url_path='/static'    # Default: '/static'
)

# Custom static folder
app = Flask(__name__, 
    static_folder='assets',
    static_url_path='/assets'
)

# Generate static file URLs
with app.test_request_context():
    print(url_for('static', filename='css/style.css'))
    # Output: /static/css/style.css
```

### Use in Templates

```html
<!DOCTYPE html>
<html>
<head>
    <title>My App</title>
    <!-- CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
</head>
<body>
    <!-- Image -->
    <img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
    
    <!-- JavaScript -->
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
    <script src="{{ url_for('static', filename='js/jquery.min.js') }}"></script>
    
    <!-- Favicon -->
    <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}">
</body>
</html>
```

### Serving Static Files in Production

```python
# In development, Flask serves static files automatically
# In production, use a web server like Nginx

# Nginx configuration example
"""
location /static {
    alias /path/to/your/app/static;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
"""
```

## Blueprints

### Blueprint Architecture

```
    ┌─────────────────┐
    │   Flask App     │
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ↓             ↓
┌───────────┐ ┌───────────┐
│ Blueprint │ │ Blueprint │
│   Auth    │ │   Blog    │
└─────┬─────┘ └─────┬─────┘
      │             │
      ↓             ↓
  ┌───────┐     ┌───────┐
  │Routes │     │Routes │
  │Views  │     │Views  │
  └───────┘     └───────┘
```

### Application Structure with Blueprints

```
your_app/
    ├── app.py
    ├── config.py
    ├── requirements.txt
    ├── blueprints/
    │   ├── __init__.py
    │   ├── auth/
    │   │   ├── __init__.py
    │   │   ├── routes.py
    │   │   ├── forms.py
    │   │   └── templates/
    │   │       └── auth/
    │   │           ├── login.html
    │   │           └── register.html
    │   └── blog/
    │       ├── __init__.py
    │       ├── routes.py
    │       ├── models.py
    │       └── templates/
    │           └── blog/
    │               ├── index.html
    │               └── post.html
    ├── static/
    └── templates/
        └── base.html
```

### Create a Blueprint

```python
# blueprints/auth/__init__.py
from flask import Blueprint

auth_bp = Blueprint(
    'auth',
    __name__,
    url_prefix='/auth',                    # All routes prefixed with /auth
    template_folder='templates',            # Blueprint-specific templates
    static_folder='static'                  # Blueprint-specific static files
)

from . import routes

# blueprints/auth/routes.py
from flask import render_template, redirect, url_for, flash, request
from . import auth_bp

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login
        return redirect(url_for('auth.dashboard'))
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle registration
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html')

@auth_bp.route('/logout')
def logout():
    # Handle logout
    return redirect(url_for('main.index'))

@auth_bp.route('/dashboard')
def dashboard():
    return render_template('auth/dashboard.html')
```

### Register Blueprints

```python
# app.py
from flask import Flask
from blueprints.auth import auth_bp
from blueprints.blog import blog_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Register blueprints
app.register_blueprint(auth_bp)              # Prefix: /auth
app.register_blueprint(blog_bp)              # Prefix: /blog

# Register with custom URL prefix
app.register_blueprint(auth_bp, url_prefix='/authentication')

# Register at root level
app.register_blueprint(blog_bp, url_prefix='/')

@app.route('/')
def index():
    return "Main Index"

if __name__ == '__main__':
    app.run(debug=True)
```

### Blueprint with Error Handlers

```python
# blueprints/blog/__init__.py
from flask import Blueprint, render_template

blog_bp = Blueprint('blog', __name__, url_prefix='/blog')

@blog_bp.errorhandler(404)
def blog_not_found(e):
    return render_template('blog/404.html'), 404

@blog_bp.errorhandler(500)
def blog_error(e):
    return render_template('blog/500.html'), 500

from . import routes
```

### Blueprint with Before/After Request Hooks

```python
# blueprints/auth/routes.py
from flask import g, session
from . import auth_bp

@auth_bp.before_request
def before_request():
    """Runs before each request to this blueprint"""
    g.user = session.get('user')

@auth_bp.after_request
def after_request(response):
    """Runs after each request to this blueprint"""
    response.headers['X-Blueprint'] = 'auth'
    return response

@auth_bp.teardown_request
def teardown_request(exception):
    """Runs after each request, even if there's an exception"""
    if exception:
        # Log the exception
        print(f"Exception: {exception}")
```

### URL Generation with Blueprints

```python
from flask import url_for

# Within the same blueprint
url_for('login')                    # Relative to current blueprint
url_for('.login')                   # Explicitly relative

# From different blueprint
url_for('auth.login')               # Fully qualified
url_for('blog.index')               # Different blueprint

# With parameters
url_for('blog.post', post_id=123)   # /blog/post/123

# External URLs
url_for('auth.login', _external=True)  # http://example.com/auth/login
```

## Flask Extensions

### Flask-Mail

Installation:

```bash
pip install flask-mail
```

Configuration:

```python
from flask import Flask
from flask_mail import Mail, Message
import os

app = Flask(__name__)

# Email configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')

mail = Mail(app)
```

Sending Emails:

```python
from flask import render_template
from flask_mail import Message
from threading import Thread

# Simple text email
@app.route('/send-simple')
def send_simple_email():
    msg = Message(
        subject="Test Email",
        sender="noreply@example.com",
        recipients=["user@example.com"]
    )
    msg.body = "This is a plain text email body"
    mail.send(msg)
    return "Email sent!"

# HTML email
@app.route('/send-html')
def send_html_email():
    msg = Message(
        subject="Welcome!",
        recipients=["user@example.com"]
    )
    msg.body = "This is the plain text fallback"
    msg.html = render_template('email/welcome.html', username='John')
    mail.send(msg)
    return "HTML email sent!"

# Email with attachment
@app.route('/send-attachment')
def send_attachment_email():
    msg = Message(
        subject="Document Attached",
        recipients=["user@example.com"]
    )
    msg.body = "Please find the attached document."
    
    with app.open_resource("document.pdf") as fp:
        msg.attach("document.pdf", "application/pdf", fp.read())
    
    mail.send(msg)
    return "Email with attachment sent!"

# Async email sending
def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)

@app.route('/send-async')
def send_async():
    msg = Message(
        subject="Async Email",
        recipients=["user@example.com"],
        body="This email is sent asynchronously"
    )
    Thread(target=send_async_email, args=(app, msg)).start()
    return "Email is being sent in background!"

# Bulk emails
@app.route('/send-bulk')
def send_bulk_emails():
    users = [
        {'email': 'user1@example.com', 'name': 'Alice'},
        {'email': 'user2@example.com', 'name': 'Bob'}
    ]
    
    with mail.connect() as conn:
        for user in users:
            msg = Message(
                subject="Bulk Email",
                recipients=[user['email']],
                body=f"Hello {user['name']}!"
            )
            conn.send(msg)
    
    return "Bulk emails sent!"
```

### Flask-Migrate

Installation:

```bash
pip install flask-migrate
```

Configuration:

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
```

Migration Commands:

```bash
flask db init  # Initialize the migration repository
flask db migrate -m "Initial migration"  # Create a new migration
flask db upgrade  # Apply the latest migration
flask db downgrade  # Revert to a previous migration
```

### Flask-Login

Installation:

```bash
pip install flask-login
```

Authentication Flow:

```
    ┌──────────────┐
    │ User Login   │
    │ Submit Form  │
    └──────┬───────┘
           │
           ↓
    ┌──────────────┐
    │  Validate    │
    │  Credentials │
    └──────┬───────┘
           │
      ┌────┴────┐
      │         │
   (Valid)  (Invalid)
      │         │
      ↓         ↓
┌──────────┐ ┌──────────┐
│login_user│ │  Show    │
│Session   │ │  Error   │
│Created   │ └──────────┘
└────┬─────┘
     │
     ↓
┌──────────────┐
│  Access      │
│  Protected   │
│  Routes      │
└──────────────┘
```

Configuration:

```python
from flask import Flask
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'          # Redirect unauthorized users
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
```

User Model:

```python
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    # UserMixin provides these methods:
    # - is_authenticated
    # - is_active
    # - is_anonymous
    # - get_id()
```

User Loader:

```python
from flask_login import login_manager

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Optional: handle invalid users
@login_manager.unauthorized_handler
def unauthorized():
    return "You must be logged in to access this page", 403
```

Login/Logout Views:

```python
from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

# Protect specific routes
@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('index'))
    return render_template('admin.html')
```

Using current_user:

```python
from flask_login import current_user

@app.route('/profile')
@login_required
def profile():
    # Access current logged-in user
    username = current_user.username
    email = current_user.email
    is_authenticated = current_user.is_authenticated
    
    return render_template('profile.html', user=current_user)

# In templates:
# {% if current_user.is_authenticated %}
#     <p>Welcome, {{ current_user.username }}!</p>
#     <a href="{{ url_for('logout') }}">Logout</a>
# {% else %}
#     <a href="{{ url_for('login') }}">Login</a>
# {% endif %}
```

## Testing

### Using pytest

Installation:

```bash
pip install pytest pytest-flask
```

Test Example:

```python
import pytest
from yourapp import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data
```

### Using unittest

```python
import unittest
from yourapp import app

class MyTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)
```

## Deployment

### Production Settings

*   Set `FLASK_ENV=production` to disable debug mode.
*   Use a production WSGI server (e.g., Gunicorn, uWSGI).
*   Configure your web server (e.g., Nginx, Apache) to proxy requests to the WSGI server.
*   Use a process manager (e.g., Supervisor, systemd) to manage the WSGI server.
*   Configure logging.
*   Use HTTPS.

### WSGI Servers

Gunicorn:

```bash
pip install gunicorn
gunicorn yourapp:app --bind 0.0.0.0:8000
```

uWSGI:

```bash
pip install uwsgi
uwsgi --http 0.0.0.0:8000 --module yourapp
```

### Environment Variables

Use environment variables for sensitive settings (e.g., `SECRET_KEY`, database credentials).

### Example Deployment with Gunicorn and Nginx

1.  Install Gunicorn: `pip install gunicorn`
2.  Create a WSGI entry point: `yourapp.py` (already covered)
3.  Create a systemd service file: `/etc/systemd/system/yourapp.service`

```ini
[Unit]
Description=Gunicorn instance to serve yourapp
After=network.target

[Service]
User=youruser
Group=www-data
WorkingDirectory=/path/to/your/app
ExecStart=/path/to/your/venv/bin/gunicorn --workers 3 --max-requests 500 --bind unix:/run/yourapp.sock yourapp:app

[Install]
WantedBy=multi-user.target
```

4.  Create an Nginx configuration file: `/etc/nginx/sites-available/yourapp`

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/yourapp.sock;
    }

    location /static {
        alias /path/to/your/app/static;
    }
}
```

5.  Create a symbolic link:

```bash
sudo ln -s /etc/nginx/sites-available/yourapp /etc/nginx/sites-enabled
```

6.  Restart Nginx:

```bash
sudo systemctl restart nginx
```

## Security

*   Use a strong `SECRET_KEY` and keep it secret.
*   Use HTTPS.
*   Sanitize user input to prevent XSS attacks.
*   Use parameterized queries to prevent SQL injection.
*   Use a Content Security Policy (CSP) to prevent various attacks.
*   Protect against CSRF attacks using Flask-WTF.
*   Limit file upload sizes.
*   Validate file uploads.
*   Use a security linter (e.g., Bandit).

## Logging

### Configure Logging

```python
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In your code:
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
```

### Logging to a File

```python
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_handler = logging.handlers.RotatingFileHandler('yourapp.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(log_handler)
```

## Flask CLI

Flask provides a command-line interface for managing your application.

*   `flask run`: Runs the development server.
*   `flask shell`: Opens a Python shell with the Flask application context.
*   `flask routes`: Shows the registered routes.
*   `flask db`: Manages database migrations (requires Flask-Migrate).

## Context Processors

Context processors inject variables automatically into all templates.

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.context_processor
def inject_variables():
    return dict(site_name="My Awesome Website")

@app.route('/')
def index():
    return render_template('index.html')
```

In `templates/index.html`:

```html
<h1>Welcome to {{ site_name }}!</h1>
```

## Error Handling

### Custom Error Pages

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
```

### Logging Exceptions

```python
import logging
from flask import Flask, render_template

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    try:
        # Some code that might raise an exception
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.exception("An error occurred")
        return render_template('error.html', error=str(e)), 500
```

## Flask-RESTful

### Installation

```bash
pip install flask-restful
```

### Define Resources

```python
from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
```

### Request Parsing

```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', required=True, help="Name is required")

class MyResource(Resource):
    def post(self):
        args = parser.parse_args()
        name = args['name']
        return {'message': f'Hello, {name}!'}
```

## Session Management

```python
from flask import Flask, session, redirect, url_for, escape, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/')
def index():
    if 'username' in session:
        return f'Logged in as {escape(session["username"])}
Click here to <a href="{url_for("logout")}">logout</a>'
    return 'You are not logged in
Click here to <a href="{url_for("login")}">login</a>'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))
```

## Flask-CORS

### Installation

```bash
pip install flask-cors
```

### Usage

```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/api/data")
def get_data():
    return {"message": "This is CORS enabled!"}
```

## Testing

### Using pytest

Installation:

```bash
pip install pytest pytest-flask
```

Test Example:

```python
import pytest
from yourapp import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data
```

### Using unittest

```python
import unittest
from yourapp import app

class MyTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)
```

## Deployment

### Production Settings

*   Set `FLASK_ENV=production` to disable debug mode.
*   Use a production WSGI server (e.g., Gunicorn, uWSGI).
*   Configure your web server (e.g., Nginx, Apache) to proxy requests to the WSGI server.
*   Use a process manager (e.g., Supervisor, systemd) to manage the WSGI server.
*   Configure logging.
*   Use HTTPS.

### WSGI Servers

Gunicorn:

```bash
pip install gunicorn
gunicorn yourapp:app --bind 0.0.0.0:8000
```

uWSGI:

```bash
pip install uwsgi
uwsgi --http 0.0.0.0:8000 --module yourapp
```

### Environment Variables

Use environment variables for sensitive settings (e.g., `SECRET_KEY`, database credentials).

### Example Deployment with Gunicorn and Nginx

1.  Install Gunicorn: `pip install gunicorn`
2.  Create a WSGI entry point: `yourapp.py` (already covered)
3.  Create a systemd service file: `/etc/systemd/system/yourapp.service`

```ini
[Unit]
Description=Gunicorn instance to serve yourapp
After=network.target

[Service]
User=youruser
Group=www-data
WorkingDirectory=/path/to/your/app
ExecStart=/path/to/your/venv/bin/gunicorn --workers 3 --max-requests 500 --bind unix:/run/yourapp.sock yourapp:app

[Install]
WantedBy=multi-user.target
```

4.  Create an Nginx configuration file: `/etc/nginx/sites-available/yourapp`

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/yourapp.sock;
    }

    location /static {
        alias /path/to/your/app/static;
    }
}
```

5.  Create a symbolic link:

```bash
sudo ln -s /etc/nginx/sites-available/yourapp /etc/nginx/sites-enabled
```

6.  Restart Nginx:

```bash
sudo systemctl restart nginx
```

## Security

*   Use a strong `SECRET_KEY` and keep it secret.
*   Use HTTPS.
*   Sanitize user input to prevent XSS attacks.
*   Use parameterized queries to prevent SQL injection.
*   Use a Content Security Policy (CSP) to prevent various attacks.
*   Protect against CSRF attacks using Flask-WTF.
*   Limit file upload sizes.
*   Validate file uploads.
*   Use a security linter (e.g., Bandit).

## Logging

### Configure Logging

```python
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In your code:
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
```

### Logging to a File

```python
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_handler = logging.handlers.RotatingFileHandler('yourapp.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(log_handler)
```

## Flask CLI

Flask provides a command-line interface for managing your application.

*   `flask run`: Runs the development server.
*   `flask shell`: Opens a Python shell with the Flask application context.
*   `flask routes`: Shows the registered routes.
*   `flask db`: Manages database migrations (requires Flask-Migrate).

To use the Flask CLI, you need to set the `FLASK_APP` environment variable:

```bash
export FLASK_APP=yourapp.py
```

Then, you can use the `flask` command:

```bash
flask run
```

## Context Processors

Context processors inject variables automatically
```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/api/data")
def get_data():
    return {"message": "This is CORS enabled!"}
```

## Testing

### Using pytest

Installation:

```bash
pip install pytest pytest-flask
```

Test Example:

```python
import pytest
from yourapp import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data
```

### Using unittest

```python
import unittest
from yourapp import app

class MyTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)
```

## Deployment

### Production Settings

*   Set `FLASK_ENV=production` to disable debug mode.
*   Use a production WSGI server (e.g., Gunicorn, uWSGI).
*   Configure your web server (e.g., Nginx, Apache) to proxy requests to the WSGI server.
*   Use a process manager (e.g., Supervisor, systemd) to manage the WSGI server.
*   Configure logging.
*   Use HTTPS.

### WSGI Servers

Gunicorn:

```bash
pip install gunicorn
gunicorn yourapp:app --bind 0.0.0.0:8000
```

uWSGI:

```bash
pip install uwsgi
uwsgi --http 0.0.0.0:8000 --module yourapp
```

### Environment Variables

Use environment variables for sensitive settings (e.g., `SECRET_KEY`, database credentials).

### Example Deployment with Gunicorn and Nginx

1.  Install Gunicorn: `pip install gunicorn`
2.  Create a WSGI entry point: `yourapp.py` (already covered)
3.  Create a systemd service file: `/etc/systemd/system/yourapp.service`

```ini
[Unit]
Description=Gunicorn instance to serve yourapp
After=network.target

[Service]
User=youruser
Group=www-data
WorkingDirectory=/path/to/your/app
ExecStart=/path/to/your/venv/bin/gunicorn --workers 3 --max-requests 500 --bind unix:/run/yourapp.sock yourapp:app

[Install]
WantedBy=multi-user.target
```

4.  Create an Nginx configuration file: `/etc/nginx/sites-available/yourapp`

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/yourapp.sock;
    }

    location /static {
        alias /path/to/your/app/static;
    }
}
```

5.  Create a symbolic link:

```bash
sudo ln -s /etc/nginx/sites-available/yourapp /etc/nginx/sites-enabled
```

6.  Restart Nginx:

```bash
sudo systemctl restart nginx
```

## Security

*   Use a strong `SECRET_KEY` and keep it secret.
*   Use HTTPS.
*   Sanitize user input to prevent XSS attacks.
*   Use parameterized queries to prevent SQL injection.
*   Use a Content Security Policy (CSP) to prevent various attacks.
*   Protect against CSRF attacks using Flask-WTF.
*   Limit file upload sizes.
*   Validate file uploads.
*   Use a security linter (e.g., Bandit).

## Logging

### Configure Logging

```python
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In your code:
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
```

### Logging to a File

```python
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_handler = logging.handlers.RotatingFileHandler('yourapp.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(log_handler)
```

## Flask CLI

Flask provides a command-line interface for managing your application.

*   `flask run`: Runs the development server.
*   `flask shell`: Opens a Python shell with the Flask application context.
*   `flask routes`: Shows the registered routes.
*   `flask db`: Manages database migrations (requires Flask-Migrate).

To use the Flask CLI, you need to set the `FLASK_APP` environment variable:

```bash
export FLASK_APP=yourapp.py
```

Then, you can use the `flask` command:

```bash
flask run
```

## Context Processors

Context processors inject variables automatically into all templates.

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.context_processor
def inject_variables():
    return dict(site_name="My Awesome Website")

@app.route('/')
def index():
    return render_template('index.html')
```

In `templates/index.html`:

```html
<h1>Welcome to {{ site_name }}!</h1>
```

## Error Handling

### Custom Error Pages

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
```

### Logging Exceptions

```python
import logging
from flask import Flask, render_template

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    try:
        # Some code that might raise an exception
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.exception("An error occurred")
        return render_template('error.html', error=str(e)), 500
```

## Flask-RESTful

### Installation

```bash
pip install flask-restful
```

### Define Resources

```python
from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
```

### Request Parsing

```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', required=True, help="Name is required")

class MyResource(Resource):
    def post(self):
        args = parser.parse_args()
        name = args['name']
        return {'message': f'Hello, {name}!'}
```

## Session Management

```python
from flask import Flask, session, redirect, url_for, escape, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/')
def index():
    if 'username' in session:
        return f'Logged in as {escape(session["username"])}
Click here to <a href="{url_for("logout")}">logout</a>'
    return 'You are not logged in
Click here to <a href="{url_for("login")}">login</a>'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))
```

## Flask-CORS

### Installation

```bash
pip install flask-cors
```

### Usage

```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/api/data")
def get_data():
    return {"message": "This is CORS enabled!"}
```

## Signals

Flask doesn't have built-in signals like Django, but you can use a third-party library like `blinker` to implement signals.

### Installation

```bash
pip install blinker
```

### Usage

```python
from flask import Flask
from blinker import signal

app = Flask(__name__)

before_request = signal('before_request')

@app.before_request
def before_request_handler():
    before_request.send(app)

@before_request.connect
def my_listener(sender):
    print("Before request signal received")

@app.route('/')
def index():
    return "Hello, World!"
```

## Flask-Limiter

### Installation

```bash
pip install Flask-Limiter
```

### Usage

```python
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/slow")
@limiter.limit("10 per minute")
def slow():
    return "Slow route"

@app.route("/fast")
def fast():
    return "Fast route"
```

## Flask-APScheduler

### Installation

```bash
pip install flask-apscheduler
```

### Usage

```python
from flask import Flask
from flask_apscheduler import APScheduler
import time

class Config(object):
    JOBS = [
        {
            'id': 'job1',
            'func': 'yourapp:job1',
            'trigger': 'interval',
            'seconds': 10
        }
    ]
    SCHEDULER_API_ENABLED = True

app = Flask(__name__)
app.config.from_object(Config())

scheduler = APScheduler()
# it is also possible to enable the API directly
# scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

def job1():
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    app.run(debug=True)
```

## Flask-Sitemap

### Installation

```bash
pip install Flask-Sitemap
```

### Usage

```python
from flask import Flask
from flask_sitemap import Sitemap

app = Flask(__name__)
ext = Sitemap(app=app)

@app.route("/sitemap.xml")
def sitemap():
    return ext.generate(base_url='http://example.com')
```

## Flask-WTF CSRF Protection

### Configuration

```python
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
csrf = CSRFProtect(app)
```

### Usage in Templates

```html
<form method="post">
    {{ form.csrf_token }}
    <!-- Your form fields -->
</form>
```

## Flask-FlatPages

### Installation

```bash
pip install Flask-FlatPages
```

### Configuration

```python
from flask import Flask
from flask_flatpages import FlatPages

app = Flask(__name__)
app.config['FLATPAGES_EXTENSION'] = '.md'
app.config['FLATPAGES_ROOT'] = 'pages'
pages = FlatPages(app)
```

### Usage

Create a directory named `pages` in your project root. Add your flat pages as `.md` files.

```python
from flask import Flask, render_template
from flask_flatpages import FlatPages, pygments_style_defs

app = Flask(__name__)
app.config['FLATPAGES_EXTENSION'] = '.md'
app.config['FLATPAGES_ROOT'] = 'pages'
app.config['FLATPAGES_MARKDOWN_EXTENSIONS'] = ['codehilite', 'fenced_code']
app.config['PYGMENTS_STYLE'] = 'default'
pages = FlatPages(app)

@app.route('/page/<path:path>')
def page(path):
    page = pages.get_or_404(path)
    return render_template('page.html', page=page, pygments_style=pygments_style_defs())
```

## Flask-Assets

### Installation

```bash
pip install Flask-Assets
```

### Configuration

```python
from flask import Flask
from flask_assets import Environment, Bundle

app = Flask(__name__)
assets = Environment(app)

js = Bundle('js/jquery.js', 'js/base.js', filters='jsmin', output='gen/packed.js')
css = Bundle('css/base.css', 'css/common.css', filters='cssmin', output='gen/all.css')

assets.register('all_js', js)
assets.register('all_css', css)
```

### Usage in Templates

```html
{% assets "all_js" %}
    <script type="text/javascript" src="{{ ASSET_URL }}"></script>
{% endassets %}

{% assets "all_css" %}
    <link rel="stylesheet" type="text/css" href="{{ ASSET_URL }}">
{% endassets %}
```

## Flask-Babel

### Installation

```bash
pip install Flask-Babel
```

### Configuration

```python
from flask import Flask
from flask_babel import Babel

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(app)
```

### Usage

```python
from flask import Flask, render_template
from flask_babel import Babel, gettext

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(app)

@app.route('/')
def index():
    title = gettext('Welcome')
    return render_template('index.html', title=title)
```

In `templates/index.html`:

```html
<h1>{{ title }}</h1>
```

## Flask-SocketIO

### Installation

```bash
pip install flask-socketio
```

### Usage

```python
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected!'})

@socketio.on('my event')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
    socketio.emit('my response', json)

if __name__ == '__main__':
    socketio.run(app, debug=True)
```

In `templates/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Flask-SocketIO Test</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js" integrity="sha512-q/dWj3kcmNeAqFvv3EY9JJ/KEvVcjtgJBmWsGGHa+YwdlOfjoOvozUvCpJlPzl5lwCDsLQIY9Mq1v8XtZiuCQ==" crossorigin="anonymous"></script>
    <script type="text/javascript" charset="utf-8">
        $(document).ready(function() {
            var socket = io();
            socket.on('connect', function() {
                socket.emit('my event', {data: 'I\'m connected!'});
            });
            socket.on('my response', function(msg) {
                $('#log').append('<p>Received: ' + msg.data + '</p>');
            });
            $('form#emit').submit(function(event) {
                socket.emit('my event', {data: $('#emit_data').val()});
                return false;
            });
        });
    </script>
</head>
<body>
    <h1>Flask-SocketIO Test</h1>
    <div id="log"></div>
    <form id="emit" method="POST" action="#">
        <input type="text" id="emit_data" name="emit_data" placeholder="Message">
        <input type="submit" value="Echo">
    </form>
</body>
</html>
```

## Flask-Principal

### Installation

```bash
pip install Flask-Principal
```

### Usage

```python
from flask import Flask, g
from flask_principal import Principal, Permission, RoleNeed, UserNeed, identity_loaded, UserContext, Identity, AnonymousIdentity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

principals = Principal(app)

# Define Needs
admin_permission = Permission(RoleNeed('admin'))
poster_permission = Permission(RoleNeed('poster'))

# Define Roles
admin_role = RoleNeed('admin')
poster_role = RoleNeed('poster')
user_need = UserNeed(1)

@identity_loaded.connect_via(app)
def on_identity_loaded(sender, identity):
    # Set the identity user object
    identity.user = get_user()

    # Add the UserNeed to the identity
    identity.provides.add(UserNeed(identity.user.id))

    # Assuming the user has a method that returns a list of roles
    for role in identity.user.roles:
        identity.provides.add(RoleNeed(role.name))

def get_user():
    # Replace with your user loading logic (e.g., from database)
    class User(object):
        def __init__(self, id, roles):
            self.id = id
            self.roles = roles

    class Role(object):
        def __init__(self, name):
            self.name = name

    admin_role = Role('admin')
    poster_role = Role('poster')

    user = User(1, [admin_role, poster_role])
    return user

@app.route('/')
def index():
    with UserContext(Identity(1)):
        if admin_permission.can():
            return "Admin access granted"
        elif poster_permission.can():
            return "Poster access granted"
        else:
            return "Access denied"

if __name__ == '__main__':
    app.run(debug=True)
```

## Flask-JWT-Extended

### Installation

```bash
pip install Flask-JWT-Extended
```

### Usage

```python
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "super-secret"  # Change this!
jwt = JWTManager(app)

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if username != "test" or password != "test":
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200
```

## Flask-Uploads

### Installation

```bash
pip install Flask-Uploads
```

### Usage

```python
from flask import Flask, request, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['SECRET_KEY'] = 'super secret key'
photos = UploadSet('photos', IMAGES)

configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        url = photos.url(filename)
        return render_template('upload.html', filename=filename, url=url)
    return render_template('upload.html')
```

In `templates/upload.html`:

```html
<!doctype html>
<html>
<head>
    <title>Upload</title>
</head>
<body>
    {% if filename %}
        <img src="{{ url }}" alt="Uploaded Image">
    {% else %}
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="photo">
            <button type="submit">Upload</button>
        </form>
    {% endif %}
</body>
</html>
```

## Flask-Mail

### Installation

```bash
pip install flask-mail
```

### Configuration

```python
from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
mail = Mail(app)
```

### Sending Emails

```python
from flask import Flask, render_template
from flask_mail import Mail, Message

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
mail = Mail(app)

@app.route('/send')
def send_email():
    msg = Message("Hello",
                  sender="your_email@gmail.com",
                  recipients=["recipient@example.com"])
    msg.body = "Hello Flask message sent from Flask-Mail"
    mail.send(msg)
    return "Sent"
```

## Flask-APScheduler

### Installation

```bash
pip install flask-apscheduler
```

### Usage

```python
from flask import Flask
from flask_apscheduler import APScheduler
import time

class Config(object):
    JOBS = [
        {
            'id': 'job1',
            'func': 'yourapp:job1',
            'trigger': 'interval',
            'seconds': 10
        }
    ]
    SCHEDULER_API_ENABLED = True

app = Flask(__name__)
app.config.from_object(Config())

scheduler = APScheduler()
# it is also possible to enable the API directly
# scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

def job1():
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    app.run(debug=True)
```

## Flask-Sitemap

### Installation

```bash
pip install Flask-Sitemap
```

### Usage

```python
from flask import Flask
from flask_sitemap import Sitemap

app = Flask(__name__)
ext = Sitemap(app=app)

@app.route("/sitemap.xml")
def sitemap():
    return ext.generate(base_url='http://example.com')
```

## Flask-WTF CSRF Protection

### Configuration

```python
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
csrf = CSRFProtect(app)
```

### Usage in Templates

```html
<form method="post">
    {{ form.csrf_token }}
    <!-- Your form fields -->
</form>
```

## Tips and Best Practices

*   Use virtual environments to isolate project dependencies.
*   Keep `SECRET_KEY` secure and out of your codebase. Use environment variables.
*   Use meaningful names for routes, variables, and functions.
*   Follow the DRY (Don't Repeat Yourself) principle.
*   Write unit tests to ensure code quality.
*   Use a production-ready web server (e.g., Gunicorn, uWSGI) and a process manager (e.g., Supervisor, systemd) for deployment.
*   Use a linter (like `flake8`) and formatter (like `black`) to ensure consistent code style.
*   Keep your code modular and reusable.
*   Document your code.
*   Use a version control system (e.g., Git).
*   Follow Flask's coding style guidelines.
*   Use Flask's built-in session management or a more robust solution like Flask-Session.
*   Monitor your application for errors and performance issues.
*   Use a CDN (Content Delivery Network) for static files.
*   Optimize database queries.
*   Use asynchronous tasks for long-running operations (e.g., sending emails) using Celery or similar.
*   Implement proper logging and error handling.
*   Regularly update Flask and its dependencies.
*   Use a security scanner to identify potential vulnerabilities.
*   Follow security best practices.
*   Use a reverse proxy like Nginx or Apache in front of your WSGI server.
*   Use a load balancer for high availability.
*   Automate deployments using tools like Fabric or Ansible.
*   Use a monitoring tool like Sentry or New Relic.
*   Implement health checks for your application.
*   Use a CDN for static assets.
*   Cache frequently accessed data.
*   Use a database connection pool.
*   Optimize your database queries.
*   Use a task queue for long-running tasks.
*   Use a background worker for asynchronous tasks.
*   Use a message queue for inter-process communication.
*   Use a service discovery tool for microservices.
*   Use a containerization tool like Docker.
*   Use an orchestration tool like Kubernetes.
