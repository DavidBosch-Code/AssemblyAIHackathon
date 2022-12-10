from dotenv import load_dotenv
import os

load_dotenv()

class ApplicationConfig():
    API_KEY = os.environ["API_KEY"]
    SQLALCHELY_ECHO = True 
    SQLALCHEMY_DATABASE_URI =  "sqlite:///project.db"
    CORS_HEADERS = "Content-Type"


headers = { } # place hearders for assembly AI here

mainUrl = "" #mainUrl for API calls