from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import SessionLocal
from app.models import User
import os

auth_bp = Blueprint('auth', __name__)

# Login Manager Setup
login_manager = LoginManager()
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    session = SessionLocal()
    user = session.query(User).get(int(user_id))
    session.close()
    return user

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Strict email validation
        if email != 'manaia.casape@gmail.com':
            flash('Access denied. Only specific email is allowed.', 'error')
            return redirect(url_for('auth.login'))
        
        session = SessionLocal()
        user = User.get_by_email(session, email)
        
        if user and user.check_password(password):
            login_user(user)
            session.close()
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
        
        session.close()
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

def init_first_user():
    """Initialize the first user if not exists"""
    session = SessionLocal()
    existing_user = User.get_by_email(session, 'manaia.casape@gmail.com')
    
    if not existing_user:
        # Create the first user with a secure initial password
        new_user = User(email='manaia.casape@gmail.com')
        new_user.set_password('Manaia2025!')  # Secure initial password
        session.add(new_user)
        session.commit()
    
    session.close()
