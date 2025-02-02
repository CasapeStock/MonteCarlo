import os
import csv
import sqlite3

def verify_csv_and_db():
    # CSV path
    csv_path = os.path.join(os.path.dirname(__file__), 'profiles.csv')
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        profiles = [row[0] for row in reader if row and row[0] and row[0].strip() and row[0] != 'profiles']
    
    print(f"Total profiles in CSV: {len(profiles)}")
    print("First 10 profiles:", profiles[:10])
    
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), 'stock_control.db')
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check existing profiles
    cursor.execute('SELECT COUNT(name) FROM steel_profiles')
    existing_count = cursor.fetchone()[0]
    print(f"Existing profiles in database: {existing_count}")
    
    # Try manual insert
    try:
        cursor.executemany(
            'INSERT OR IGNORE INTO steel_profiles (name) VALUES (?)', 
            [(profile,) for profile in profiles]
        )
        conn.commit()
        print(f"Inserted {cursor.rowcount} new profiles")
    except Exception as e:
        print(f"Error inserting profiles: {e}")
    
    conn.close()

if __name__ == '__main__':
    verify_csv_and_db()
