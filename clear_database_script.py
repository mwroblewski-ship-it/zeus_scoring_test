#!/usr/bin/env python3
"""
Script do wyczyszczenia challenges database żeby usunąć stare precipitation challenges
"""

import sqlite3
import os
from pathlib import Path

def clear_challenges_database():
    """
    Wyczyść bazę danych challenges ze starymi precipitation challenges
    """
    
    # Standardowa ścieżka do bazy
    db_path = Path.home() / ".cache" / "zeus" / "challenges.db"
    
    print(f"🗄️ Clearing challenges database...")
    print(f"Database path: {db_path}")
    
    if not db_path.exists():
        print(f"❌ Database file doesn't exist: {db_path}")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Sprawdź ile challenges mamy
            cursor.execute("SELECT COUNT(*) FROM challenges")
            total_challenges = cursor.fetchone()[0]
            
            # Sprawdź ile precipitation challenges
            cursor.execute("SELECT COUNT(*) FROM challenges WHERE variable = 'total_precipitation'")
            precipitation_challenges = cursor.fetchone()[0]
            
            print(f"📊 Current database status:")
            print(f"   Total challenges: {total_challenges}")
            print(f"   Precipitation challenges: {precipitation_challenges}")
            
            if precipitation_challenges > 0:
                print(f"🗑️ Deleting precipitation challenges...")
                
                # Usuń responses dla precipitation challenges
                cursor.execute("""
                    DELETE FROM responses 
                    WHERE challenge_uid IN (
                        SELECT uid FROM challenges WHERE variable = 'total_precipitation'
                    )
                """)
                responses_deleted = cursor.rowcount
                
                # Usuń precipitation challenges
                cursor.execute("DELETE FROM challenges WHERE variable = 'total_precipitation'")
                challenges_deleted = cursor.rowcount
                
                conn.commit()
                
                print(f"✅ Deleted:")
                print(f"   Challenges: {challenges_deleted}")
                print(f"   Responses: {responses_deleted}")
            else:
                print(f"✅ No precipitation challenges to delete")
            
            # Sprawdź końcowy stan
            cursor.execute("SELECT COUNT(*) FROM challenges")
            remaining_challenges = cursor.fetchone()[0]
            
            cursor.execute("SELECT DISTINCT variable FROM challenges")
            remaining_variables = [row[0] for row in cursor.fetchall()]
            
            print(f"📊 Final database status:")
            print(f"   Remaining challenges: {remaining_challenges}")
            print(f"   Remaining variables: {remaining_variables}")
            
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def clear_entire_database():
    """
    Usuń całą bazę danych (nuclear option)
    """
    db_path = Path.home() / ".cache" / "zeus" / "challenges.db"
    
    print(f"💣 NUCLEAR OPTION: Deleting entire database...")
    print(f"Database path: {db_path}")
    
    if db_path.exists():
        os.remove(db_path)
        print(f"✅ Database deleted successfully")
    else:
        print(f"❌ Database file doesn't exist")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--nuclear":
        clear_entire_database()
    else:
        clear_challenges_database()
        print(f"\n💡 If problems persist, run with --nuclear to delete entire database")