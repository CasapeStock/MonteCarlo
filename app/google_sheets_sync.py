import logging
from datetime import datetime
from database import SessionLocal
from app.models import StockEntry

class GoogleSheetsSynchronizer:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Google Sheets Synchronizer initialized without external connection")
    
    def sync_to_google_sheets(self, entries):
        """
        Placeholder method for syncing entries to Google Sheets
        Currently just logs the entries that would be synced
        """
        try:
            data_to_sync = [
                {
                    'ID': entry.id, 
                    'Date': entry.date.strftime('%Y-%m-%d'),
                    'Profile': entry.profile,
                    'Quantity': entry.quantity,
                    'Length': entry.length,
                    'Steel Type': entry.steel_type,
                    'Stock Reference': entry.stock_reference,
                    'Work Number': entry.work_number
                } for entry in entries
            ]
            
            self.logger.info(f"Prepared {len(data_to_sync)} entries for sync")
            return data_to_sync
        
        except Exception as e:
            self.logger.error(f"Sync preparation error: {e}")
            return []
    
    def sync_from_google_sheets(self, session):
        """
        Placeholder method for syncing data from Google Sheets to local database
        Currently does nothing
        """
        self.logger.info("Sync from Google Sheets is currently disabled")
        return []
