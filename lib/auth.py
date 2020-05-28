import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow


def google_spreadsheet_auth(
        credentials_path=os.path.join(os.path.dirname(__file__),'../credentials.json'), 
        token_path=os.path.join(os.path.dirname(__file__),'../token.pickle')):
    SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    else:
        creds = None
    # If there are no (valid) credentials available, let the user log in.
    if creds is None or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    service = build('sheets', 'v4', credentials=creds).spreadsheets()
    return service

if __name__ == "__main__":
    google_spreadsheet_auth()