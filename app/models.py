from sqlalchemy import Column, Integer, String, Float, Date, func, Boolean
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

Base = declarative_base()

class User(UserMixin, Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @classmethod
    def get_by_email(cls, session, email):
        return session.query(cls).filter_by(email=email).first()

class SteelProfile(Base):
    __tablename__ = 'steel_profiles'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    profile_type = Column(String(50), nullable=True)  # e.g., HEA, HEB, IPE, RHS

    @classmethod
    def bulk_insert_profiles(cls, session, profiles):
        """
        Bulk insert profiles, avoiding duplicates
        :param session: SQLAlchemy session
        :param profiles: List of profile names
        """
        existing_profiles = {p.name for p in session.query(cls).all()}
        new_profiles = [
            cls(name=profile) for profile in profiles 
            if profile and profile not in existing_profiles
        ]
        
        if new_profiles:
            session.add_all(new_profiles)
            session.commit()
        
        return len(new_profiles)

class StockEntry(Base):
    __tablename__ = 'stock_entries'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, default=datetime.now)
    profile = Column(String(100), nullable=False)
    quantity = Column(Integer, nullable=False)
    length = Column(Float, nullable=False)
    steel_type = Column(String(100), nullable=False)
    stock_reference = Column(String(100), nullable=False)
    work_number = Column(String(100), nullable=False)

    @classmethod
    def filter_entries(cls, session, **kwargs):
        """Advanced filtering method"""
        query = session.query(cls)
        
        # Date range filter
        if 'start_date' in kwargs and 'end_date' in kwargs:
            query = query.filter(
                cls.date.between(
                    datetime.strptime(kwargs['start_date'], '%Y-%m-%d'),
                    datetime.strptime(kwargs['end_date'], '%Y-%m-%d')
                )
            )
        
        # Exact match filters
        exact_match_fields = [
            'profile', 'steel_type', 'stock_reference', 'work_number'
        ]
        for field in exact_match_fields:
            if field in kwargs:
                query = query.filter(getattr(cls, field) == kwargs[field])
        
        # Range filters
        if 'min_quantity' in kwargs:
            query = query.filter(cls.quantity >= int(kwargs['min_quantity']))
        if 'max_quantity' in kwargs:
            query = query.filter(cls.quantity <= int(kwargs['max_quantity']))
        
        if 'min_length' in kwargs:
            query = query.filter(cls.length >= float(kwargs['min_length']))
        if 'max_length' in kwargs:
            query = query.filter(cls.length <= float(kwargs['max_length']))
        
        # Sorting
        if 'sort_by' in kwargs:
            sort_field = getattr(cls, kwargs['sort_by'], None)
            if sort_field is not None:
                query = query.order_by(sort_field)
        
        return query.all()

    @classmethod
    def get_summary_stats(cls, session):
        """Generate summary statistics"""
        return {
            'total_entries': session.query(cls).count(),
            'total_quantity': session.query(func.sum(cls.quantity)).scalar() or 0,
            'unique_steel_types': session.query(cls.steel_type.distinct()).count(),
            'unique_profiles': session.query(cls.profile.distinct()).count(),
        }
