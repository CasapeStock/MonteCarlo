import os
import sys
import csv
import sqlite3

def import_profiles_from_csv(csv_path):
    # Determine the database path
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create steel_profiles table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS steel_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    try:
        # Read profiles from CSV
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            profiles = [row[0] for row in reader if row and row[0] != 'profiles']
        
        # Remove duplicates while preserving order
        unique_profiles = []
        seen = set()
        for profile in profiles:
            if profile not in seen:
                unique_profiles.append(profile)
                seen.add(profile)
        
        # Insert new profiles, ignoring duplicates
        inserted_count = 0
        for profile in unique_profiles:
            try:
                cursor.execute('INSERT OR IGNORE INTO steel_profiles (name) VALUES (?)', (profile,))
                if cursor.rowcount > 0:
                    inserted_count += 1
            except sqlite3.IntegrityError:
                # Skip if profile already exists
                pass
        
        # Commit changes
        conn.commit()
        
        print(f"Successfully imported {inserted_count} new profiles.")
    
    except Exception as e:
        print(f"Error importing profiles: {e}")
    
    finally:
        conn.close()

if __name__ == '__main__':
    import_profiles_from_csv('profiles.csv')
