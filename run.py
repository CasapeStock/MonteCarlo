from flask import Flask, render_template, request, redirect, url_for, flash, session, g
import os
import sqlite3
from datetime import datetime, date
import re

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secure random secret key
MAX_UNDO_ENTRIES = 5  # Maximum number of entries that can be undone

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('stock_control.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_cached_db_connection():
    """
    Get a cached database connection to reduce connection overhead
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'stock_control.db'))
        g.sqlite_db.row_factory = sqlite3.Row
    return g.sqlite_db

@app.teardown_appcontext
def close_db(error):
    """
    Close the database connection at the end of the request
    """
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

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

def group_and_limit_profiles(profiles, max_profiles=500):
    """
    Group profiles by series and limit total number of profiles
    Prioritizes most common profile series
    """
    # Define priority order for profile series
    priority_series = ['HEA', 'HEB', 'HEM', 'IPE', 'B', 'UNP']
    
    # Group profiles by series
    profile_groups = {}
    for profile in profiles:
        # Extract series prefix
        series = ''.join([char for char in profile if char.isalpha()])
        
        if series not in profile_groups:
            profile_groups[series] = []
        profile_groups[series].append(profile)
    
    # Sort groups by priority and size
    sorted_groups = sorted(
        profile_groups.items(), 
        key=lambda x: (
            priority_series.index(x[0]) if x[0] in priority_series else len(priority_series), 
            -len(x[1])
        )
    )
    
    # Collect profiles, respecting max limit
    selected_profiles = []
    for series, group_profiles in sorted_groups:
        # Sort profiles within group
        group_profiles.sort()
        
        # Add profiles from this group
        for profile in group_profiles:
            if len(selected_profiles) < max_profiles:
                selected_profiles.append(profile)
            else:
                break
        
        if len(selected_profiles) >= max_profiles:
            break
    
    return selected_profiles

def read_profiles_from_csv(csv_path, custom_path=None, use_cache=True):
    """
    Read profiles directly from CSV file, with custom profile prioritization
    and caching to improve performance
    """
    # Cache file to store processed profiles
    cache_path = os.path.join(os.path.dirname(__file__), 'profiles_cache.txt')
    
    # Check if cache exists and is recent
    if use_cache and os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        csv_mtime = os.path.getmtime(csv_path)
        custom_mtime = os.path.getmtime(custom_path) if custom_path and os.path.exists(custom_path) else 0
        
        # If cache is newer than both CSV files, use cached profiles
        if cache_mtime > csv_mtime and cache_mtime > custom_mtime:
            try:
                with open(cache_path, 'r', encoding='utf-8') as cache_file:
                    return [line.strip() for line in cache_file if line.strip()]
            except Exception:
                pass
    
    profiles = []
    
    # Read custom profiles first if custom path is provided
    if custom_path and os.path.exists(custom_path):
        try:
            with open(custom_path, 'r', encoding='utf-8') as custom_csvfile:
                custom_lines = [line.strip() for line in custom_csvfile if line.strip()]
                
                for line in custom_lines:
                    # Clean the custom profile
                    cleaned_profile = clean_profile(line)
                    if cleaned_profile:
                        profiles.append(cleaned_profile)
        except Exception:
            pass
    
    # Read main profiles
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Read all lines and strip whitespace
            lines = [line.strip() for line in csvfile if line.strip()]
            
            print(f"Total profiles in main CSV: {len(lines)}")
            print("First 10 profiles:", lines[:10])
            
            # Process each line
            for line in lines:
                # Skip problematic lines
                if line.lower() in ['', 'total', 'count', 'sum']:
                    continue
                
                # Replace comma with dot for decimal numbers
                line = line.replace(',', '.')
                
                # Replace 'x' or 'X' with '*' for multiplication
                line = line.replace('x', '*').replace('X', '*')
                
                # Basic profile validation
                if any(char.isalpha() for char in line):
                    # Only add if not already in profiles
                    cleaned_profile = clean_profile(line)
                    if cleaned_profile and cleaned_profile not in profiles:
                        profiles.append(cleaned_profile)
        
        print(f"Total unique profiles after processing: {len(profiles)}")
        
        # Write to cache for faster future access
        try:
            with open(cache_path, 'w', encoding='utf-8') as cache_file:
                for profile in profiles:
                    cache_file.write(f"{profile}\n")
        except Exception as e:
            print(f"Error writing cache: {e}")
        
        return profiles
    
    except Exception as e:
        print(f"Error reading profiles: {e}")
        return profiles

def init_db():
    """
    Initialize the database with tables, preserving existing data
    """
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    
    # Connect to database (will create if not exists)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create steel_profiles table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS steel_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT
        )
    ''')
    
    # Create stock_entries table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            profile TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            length REAL NOT NULL,
            steel_type TEXT NOT NULL,
            stock_reference TEXT,
            work_number TEXT
        )
    ''')
    
    # Create stock_used table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_used (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_stock_id INTEGER,
            profile TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL,
            previous_work_number TEXT DEFAULT '',
            newest_work_number TEXT DEFAULT '',
            date_used DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(original_stock_id) REFERENCES stock_entries(id)
        )
    ''')
    
    # Populate steel_profiles table from CSV only if empty
    cursor.execute('SELECT COUNT(*) FROM steel_profiles')
    profile_count = cursor.fetchone()[0]
    
    if profile_count == 0:
        # CSV path for profiles
        csv_path = os.path.join(os.path.dirname(__file__), 'profiles.csv')
        custom_path = os.path.join(os.path.dirname(__file__), 'custom_profiles.csv')
        
        profiles = read_profiles_from_csv(csv_path, custom_path)
        
        print(f"Profiles to insert: {len(profiles)}")
        
        for profile in profiles:
            try:
                print(f"Inserting profile: {profile}")
                cursor.execute('INSERT OR IGNORE INTO steel_profiles (name) VALUES (?)', (profile,))
            except sqlite3.IntegrityError:
                print(f"Duplicate profile skipped: {profile}")
            except Exception as e:
                print(f"Error inserting profile {profile}: {e}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database initialized successfully.")

# Call init_db when the application starts
init_db()

@app.route('/')
def index():
    # Clear any previous undo information when landing on home page
    session.pop('deleted_entries', None)
    return render_template('index.html')

@app.route('/add_entry', methods=['GET', 'POST'])
def add_entry():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    
    # CSV path for profiles
    csv_path = os.path.join(os.path.dirname(__file__), 'profiles.csv')
    
    # Optional custom profiles path
    custom_path = os.path.join(os.path.dirname(__file__), 'custom_profiles.csv')
    
    try:
        # Connect to database
        conn = get_cached_db_connection()
        cursor = conn.cursor()
        
        # Fetch steel profiles from database
        cursor.execute('SELECT DISTINCT name FROM steel_profiles ORDER BY name')
        db_profiles = [profile[0] for profile in cursor.fetchall()]
        
        # Read profiles from CSV
        csv_profiles = read_profiles_from_csv(csv_path, custom_path)
        
        # Combine and deduplicate profiles
        steel_profiles = list(dict.fromkeys(db_profiles + csv_profiles))
        
        # Sort profiles
        steel_profiles.sort()
        
        if request.method == 'POST':
            # Handle custom profile
            profile = request.form.get('profile')
            if profile == 'custom':
                profile = request.form.get('custom_profile')
            
            # Print out all form data for debugging
            print("Form Data Received:")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            
            # Simplified entry insertion with more robust error handling
            try:
                # Validate required fields
                required_fields = ['date', 'profile', 'quantity', 'length', 'steel_type', 'stock_reference', 'work_number']
                for field in required_fields:
                    if not request.form.get(field):
                        raise ValueError(f"Missing required field: {field}")
                
                cursor.execute('''
                    INSERT INTO stock_entries 
                    (date, profile, quantity, length, steel_type, stock_reference, work_number) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request.form['date'],
                    profile,
                    int(request.form['quantity']),
                    float(request.form['length']),
                    request.form['steel_type'],
                    request.form['stock_reference'],
                    request.form['work_number']
                ))
                conn.commit()
                flash('Entry added successfully!', 'success')
                print("Entry added successfully!")
            except Exception as e:
                # Rollback the transaction
                conn.rollback()
                
                # Log the full error details
                import traceback
                print(f"Error adding entry: {str(e)}")
                print("Traceback:")
                traceback.print_exc()
                
                # Flash a more informative error message
                flash(f'Error adding entry: {str(e)}', 'error')
            finally:
                conn.close()
            
            return redirect(url_for('index'))
        
        # Pass current date and steel profiles to the template
        current_date = date.today().strftime('%Y-%m-%d')
        return render_template('add_entry.html', 
                               current_date=current_date, 
                               max_date=current_date, 
                               steel_profiles=steel_profiles)
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/filter_entries', methods=['GET', 'POST'])
def filter_entries():
    """
    Filter and display stock entries, including used stock
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get filter parameters
        steel_type = request.form.get('steel_type', '')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        profile_type = request.form.get('min_quantity', '')
        work_number = request.form.get('max_quantity', '')
        
        # Construct base query
        query = "SELECT * FROM stock_entries WHERE 1=1"
        params = []

        # Steel Type Filter
        if steel_type:
            query += " AND steel_type = ?"
            params.append(steel_type)

        # Start Date Filter
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        # End Date Filter
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        # Profile Type Filter
        if profile_type:
            query += " AND profile = ?"
            params.append(profile_type)

        # Work Number Filter
        if work_number:
            query += " AND work_number LIKE ?"
            params.append(f"%{work_number}%")

        # Execute the query
        cursor.execute(query, params)
        entries = cursor.fetchall()
        
        # Detailed logging for entries
        print("\n--- DETAILED ENTRIES LOGGING ---")
        print(f"Total Entries: {len(entries)}")
        print("Columns: id, date, profile, quantity, length, steel_type, stock_reference, work_number")
        
        # Prepare entries for template
        formatted_entries = []
        for entry in entries:
            print(f"Entry: {entry}")
            formatted_entries.append({
                'id': entry[0],
                'date': entry[1],
                'profile': entry[2],
                'quantity': entry[3],
                'length': entry[4],
                'steel_type': entry[5],
                'stock_reference': entry[6],
                'work_number': entry[7]
            })
        
        # Logging formatted entries
        print("\n--- FORMATTED ENTRIES ---")
        for formatted_entry in formatted_entries:
            print(f"Formatted Entry: {formatted_entry}")
        
        # Get steel profiles for dropdown
        cursor.execute('SELECT name FROM steel_profiles ORDER BY name')
        steel_profiles = [profile[0] for profile in cursor.fetchall()]
        
        return render_template('filter_entries.html', 
                               entries=formatted_entries, 
                               steel_profiles=steel_profiles)
    
    except Exception as e:
        # Log any errors
        print(f"Error in filter_entries: {e}")
        flash(f'Error filtering entries: {e}', 'error')
        
    finally:
        conn.close()
    
    return render_template('filter_entries.html', entries=[], steel_profiles=[])

@app.route('/delete_entry/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Fetch the full entry details
        cursor.execute('SELECT * FROM stock_entries WHERE id = ?', (entry_id,))
        entry = cursor.fetchone()
        
        if entry:
            # Delete the entry from stock_entries
            cursor.execute('DELETE FROM stock_entries WHERE id = ?', (entry_id,))
            conn.commit()
            
            # Manage the list of deleted entries in the session
            if 'deleted_entries' not in session:
                session['deleted_entries'] = []
            
            # Add the current entry to the list of deleted entries
            deleted_entry = {
                'id': entry_id,
                'date': entry['date'],
                'profile': entry['profile'],
                'quantity': entry['quantity'],
                'length': entry['length'],
                'steel_type': entry['steel_type'],
                'stock_reference': entry['stock_reference'],
                'work_number': entry['work_number']
            }
            
            # Insert at the beginning of the list
            session['deleted_entries'].insert(0, deleted_entry)
            
            # Limit to MAX_UNDO_ENTRIES
            session['deleted_entries'] = session['deleted_entries'][:MAX_UNDO_ENTRIES]
            
            # Persist the changes to the session
            session.modified = True
            
            flash(f'Entry {entry_id} deleted successfully.', 'warning')
        else:
            flash('No entry found to delete.', 'error')
    
    except Exception as e:
        flash(f'Error deleting entry: {str(e)}', 'error')
    
    finally:
        conn.close()
    
    return redirect(url_for('filter_entries'))

@app.route('/undo_delete', methods=['POST'])
def undo_delete():
    # Check if there are any deleted entries to undo
    if 'deleted_entries' not in session or len(session['deleted_entries']) == 0:
        flash('No entries to undo.', 'error')
        return redirect(url_for('filter_entries'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get the first (most recent) deleted entry
        deleted_entry = session['deleted_entries'].pop(0)
        
        # Reinsert the entry into stock_entries
        cursor.execute('''
            INSERT INTO stock_entries 
            (id, date, profile, quantity, length, steel_type, stock_reference, work_number) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deleted_entry['id'], 
            deleted_entry['date'], 
            deleted_entry['profile'], 
            deleted_entry['quantity'], 
            deleted_entry['length'], 
            deleted_entry['steel_type'], 
            deleted_entry['stock_reference'], 
            deleted_entry['work_number']
        ))
        
        conn.commit()
        
        # Update the session
        session.modified = True
        
        # Provide feedback about the restored entry
        flash(f'Entry {deleted_entry["id"]} restored successfully!', 'success')
    
    except Exception as e:
        flash(f'Error restoring entry: {str(e)}', 'error')
    
    finally:
        conn.close()
    
    return redirect(url_for('filter_entries'))

@app.route('/use_stock', methods=['POST'])
def use_stock():
    """
    Mark stock entries as used
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get form data
        profile = request.form.get('profile')
        previous_work_number = request.form.get('previous_work_number')
        newest_work_number = request.form.get('newest_work_number')
        quantity_used = float(request.form.get('quantity_used', 0))
        
        # Validate inputs
        if not profile or not newest_work_number or quantity_used <= 0:
            flash('Invalid input. Please fill all required fields.', 'error')
            return redirect(url_for('filter_entries'))
        
        # Find the most recent entry for this profile
        cursor.execute('''
            SELECT id, quantity 
            FROM stock_entries 
            WHERE profile = ? 
            ORDER BY date DESC 
            LIMIT 1
        ''', (profile,))
        
        entry = cursor.fetchone()
        
        if not entry:
            flash(f'No stock entries found for profile: {profile}', 'error')
            return redirect(url_for('filter_entries'))
        
        entry_id, available_quantity = entry
        
        # Check if quantity used is valid
        if quantity_used > available_quantity:
            flash(f'Cannot use {quantity_used}. Only {available_quantity} available.', 'error')
            return redirect(url_for('filter_entries'))
        
        # Update stock entries
        cursor.execute('''
            UPDATE stock_entries 
            SET quantity = quantity - ? 
            WHERE id = ?
        ''', (quantity_used, entry_id))
        
        # Insert into stock_used table
        cursor.execute('''
            INSERT INTO stock_used 
            (profile, previous_work_number, newest_work_number, quantity, date_used) 
            VALUES (?, ?, ?, ?, date('now'))
        ''', (profile, previous_work_number, newest_work_number, quantity_used))
        
        # Commit changes
        conn.commit()
        
        # Flash success message
        flash(f'Successfully used {quantity_used} of {profile} stock', 'success')
        
        return redirect(url_for('filter_entries'))
    
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        
        # Log the error
        print(f"Error in use_stock: {e}")
        
        # Flash error message
        flash(f'Error using stock: {e}', 'error')
        
        return redirect(url_for('filter_entries'))
    
    finally:
        # Always close the connection
        conn.close()

# Modify app configuration for better performance
app.config['DEBUG'] = True  # Enable debug mode
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reload
app.config['EXPLAIN_TEMPLATE_LOADING'] = True  # Enable template loading explanations

# Run the application
if __name__ == '__main__':
    # Ensure the database is initialized before running
    init_db()
    
    # Run the Flask application
    app.run(
        host='0.0.0.0',  # Listen on all available network interfaces
        port=5000,        # Standard Flask development port
        debug=True        # Enable debug mode for development
    )
