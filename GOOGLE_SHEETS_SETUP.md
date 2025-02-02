# Google Sheets Integration Setup

## Prerequisites
- A Google Account
- Access to Google Cloud Console
- The spreadsheet you want to use

## Step-by-Step Setup

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Google Sheets API**
   - In the Google Cloud Console, go to "APIs & Services"
   - Click "Enable APIs and Services"
   - Search for "Google Sheets API" and enable it

3. **Create a Service Account**
   - In "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in service account details
   - Generate a JSON key file
   - Download the JSON key file

4. **Share Your Spreadsheet**
   - Open your Google Sheet
   - Click "Share"
   - Add the service account email (from the JSON key) as an editor

5. **Configure the Application**
   - Rename the downloaded JSON key file to `credentials.json`
   - Place `credentials.json` in the project root directory

## Troubleshooting
- Ensure the service account email has edit access to the sheet
- Verify the JSON key file is correctly formatted
- Check that all dependencies are installed

## Security Notes
- Never commit `credentials.json` to version control
- Keep the JSON key file secure and private
