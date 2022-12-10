from dotenv import load_dotenv
import os

load_dotenv()


class ApplicationConfig():
    API_KEY = os.environ["API_KEY"]
    SQLALCHELY_ECHO = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///project.db"
    CORS_HEADERS = "Content-Type"


headers = {}  # place hearders for assembly AI here

ASSEMBLY_URL = "https://api.assemblyai.com/v2/"

ASSEMBLY_UPLOAD_ENDPOINT = ASSEMBLY_URL + 'upload'
ASSEMBLY_TRANSCRIPT_ENDPOINT = ASSEMBLY_URL + 'transcript'

API_KEY = os.environ["API_KEY"]

CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')
CSV_CACHE = os.path.join(CACHE_FOLDER, 'cache.csv')
