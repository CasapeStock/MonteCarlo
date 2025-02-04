import os
import sqlite3
import csv
import re
import sys
import traceback

def create_tables():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    
    # Ensure absolute path
    db_path = os.path.abspath(db_path)
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create steel_profiles table with more robust constraints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS steel_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create stock_entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                profile TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                length REAL,
                steel_type TEXT,
                stock_reference TEXT,
                work_number TEXT
            )
        ''')
        
        # Commit changes
        conn.commit()
        
        print(f"Database tables created successfully at {db_path}")
    
    except sqlite3.Error as e:
        print(f"Error creating tables: {e}")
        traceback.print_exc()
    
    finally:
        if 'conn' in locals():
            conn.close()

def clean_profile(profile):
    """
    Clean and validate profile names with robust parsing
    """
    if not profile:
        return None
    
    # Remove leading/trailing whitespace
    profile = profile.strip()
    
    # Replace comma with dot for decimal numbers
    profile = profile.replace(',', '.')
    
    # Replace 'x' or 'X' with '*' for multiplication
    profile = profile.replace('x', '*').replace('X', '*')
    
    # Remove any non-alphanumeric characters except '*' and '/'
    profile = re.sub(r'[^a-zA-Z0-9*/.]', '', profile)
    
    # Ensure profile has at least one letter and is not just numbers
    if not re.search(r'[a-zA-Z]', profile):
        return None
    
    return profile

def import_profiles(verbose=True):
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    db_path = os.path.abspath(db_path)
    
    # CSV path
    csv_path = os.path.join(os.path.dirname(__file__), 'profiles.csv')
    csv_path = os.path.abspath(csv_path)
    
    # Extensive logging
    if verbose:
        print(f"Database Path: {db_path}")
        print(f"CSV Path: {csv_path}")
        print(f"CSV Exists: {os.path.exists(csv_path)}")
        print(f"CSV Size: {os.path.getsize(csv_path)} bytes")
    
    # Collect unique profiles
    profiles = set()
    
    try:
        # Read CSV file
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Read all lines
            lines = [line.strip() for line in csvfile if line.strip()]
            
            # Verbose logging of raw lines
            if verbose:
                print("Raw Lines:", lines[:10])  # Print first 10 lines
            
            # Process each line
            for line in lines:
                # Skip lines that are empty or problematic
                if line.lower() in ['', 'total', 'count', 'sum']:
                    continue
                
                # Clean the profile
                cleaned_profile = clean_profile(line)
                
                # Add if valid
                if cleaned_profile:
                    profiles.add(cleaned_profile)
        
        # Convert to sorted list
        sorted_profiles = sorted(list(profiles))
        
        if verbose:
            print(f"Total Unique Profiles Found: {len(sorted_profiles)}")
            print("First 50 Profiles:", sorted_profiles[:50])
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Clear existing profiles to avoid duplicates
        cursor.execute('DELETE FROM steel_profiles')
        
        # Bulk insert profiles with detailed error handling
        try:
            cursor.executemany(
                'INSERT OR IGNORE INTO steel_profiles (name) VALUES (?)', 
                [(profile,) for profile in sorted_profiles]
            )
            
            # Verify insertion
            cursor.execute('SELECT COUNT(*) FROM steel_profiles')
            profile_count = cursor.fetchone()[0]
            
            if verbose:
                print(f"Profiles Actually Inserted: {profile_count}")
            
            conn.commit()
        
        except sqlite3.Error as insert_error:
            print(f"Database Insertion Error: {insert_error}")
            conn.rollback()
        
        finally:
            conn.close()
    
    except Exception as e:
        print(f"Unexpected Error Importing Profiles: {e}")
    
    return sorted_profiles

# Run both functions when script is executed
if __name__ == '__main__':
    create_tables()
    import_profiles(verbose=True)
