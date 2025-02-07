
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
venv\Scripts\activate  # On Windows
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
python your_app_name.py
```

## Routing

### Basic Route

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

if __name__ == '__main__':
    app.run(debug=True)
```

### Dynamic Routes

```python
from flask import Flask

app = Flask(__name__)

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {username}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'

if __name__ == '__main__':
    app.run(debug=True)
```

### HTTP Methods

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return "Do the login"
    else:
        return "Show the login form"

if __name__ == '__main__':
    app.run(debug=True)
```

### URL Building

```python
from flask import Flask, url_for

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

with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))

if __name__ == '__main__':
    app.run(debug=True)
```

## Templates

### Basic Template Rendering

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

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

*   `{{ variable }}`: Outputs a variable.
*   `{% tag %}`: Template logic tag (e.g., `for`, `if`).
*   `{{ variable|filter }}`: Applies a filter to a variable.
*   `{% extends "base.html" %}`: Extends a base template.
*   `{% block block_name %}{% endblock %}`: Defines a block for template inheritance.
*   `{% include "template_name.html" %}`: Includes another template.
*   `{% url_for 'view_name' arg1=value1 %}`: Generates a URL for a view.

### Common Template Filters

*   `safe`: Marks a string as safe for HTML output.
*   `capitalize`: Capitalizes the first character of a string.
*   `lower`, `upper`: Converts a string to lowercase or uppercase.
*   `title`: Converts a string to title case.
*   `trim`: Removes leading and trailing whitespace.
*   `striptags`: Strips SGML/XML tags.
*   `length`: Returns the length of a value.
*   `default(value, default_value='')`: Provides a default value if a variable is undefined.
*   `replace(old, new, count=None)`: Replaces occurrences of a substring.
*   `format(value, *args, **kwargs)`: Formats a string using Python's string formatting.

## Forms

### Basic Form

```html
<form method="post">
    <label for="name">Name:</label><br>
    <input type="text" id="name" name="name"><br>
    <label for="email">Email:</label><br>
    <input type="email" id="email" name="email"><br>
    <input type="submit" value="Submit">
</form>
```

### Using Flask-WTF

Installation:

```bash
pip install flask-wtf
```

Configuration (in your app):

```python
import os
from flask import Flask
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your_secret_key'
```

### Define a Form (forms.py)

```python
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, EmailField, BooleanField
from wtforms.validators import DataRequired, Length, Email

class MyForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=20)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    message = TextAreaField('Message', validators=[DataRequired()])
    agree = BooleanField('I agree to the terms', validators=[DataRequired()])
    submit = SubmitField('Submit')
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
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
```

### Define a Model

```python
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}', '{self.date_posted}')"
```

### Create and Manage Tables

```python
from yourapp import app, db

with app.app_context():
    db.create_all()  # Create tables

# In the Python shell:
# from yourapp import db, User, Post
# user_1 = User(username='Corey', email='corey@example.com')
# db.session.add(user_1)
# db.session.commit()
```

### Querying the Database

```python
from yourapp import db, User, Post

# Get all users
all_users = User.query.all()

# Filter users
filtered_users = User.query.filter_by(username='Corey')

# Get a single user by ID
user = User.query.get(1)

# Get a single user, handling 404 error
user = User.query.get_or_404(1)

# Create a new user
new_user = User(username='NewUser', email='new@example.com', password='password')
db.session.add(new_user)
db.session.commit()

# Update an existing user
user = User.query.get(1)
user.email = 'updated@example.com'
db.session.commit()

# Delete a user
user = User.query.get(1)
db.session.delete(user)
db.session.commit()

# Relationships
user = User.query.get(1)
posts = user.posts  # Access posts related to the user
```

## Static Files

### Configure Static Files

Create a `static` folder in your app directory.

In your template:

```html
{% load static %}
<link rel="stylesheet" type="text/css" href="{% static 'css/style.css' %}">
<img src="{% static 'images/logo.png' %}">
```

## Blueprints

### Create a Blueprint

```python
from flask import Blueprint

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return "Main Blueprint Index"
```

### Register a Blueprint

```python
from flask import Flask
from yourapp.main import main

app = Flask(__name__)
app.register_blueprint(main)
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

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
mail = Mail(app)
```

Sending Emails:

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

Configuration:

```python
from flask import Flask
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Specify the login view
```

User Model:

```python
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    # ... other fields ...
```

User Loader Callback:

```python
from yourapp import login_manager, User

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
```

Protecting Views:

```python
from flask_login import login_required

@app.route('/protected')
@login_required
def protected():
    return "Protected View"
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
