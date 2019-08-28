import os
import json

# importing the requests library 
import requests 

# explicitly providing path to '.env'
from dotenv import load_dotenv
env_path = os.path.join('.env')
print(env_path)
load_dotenv(dotenv_path=env_path)

# api-endpoint 
URL = os.getenv("HTML_REQUEST_URL")
print(URL)

# defining the data to be sent to the API 
DATA = json.loads('{"words": ["*"], "dates": {"startDate": "", "endDate": ""}, "searchType": "standard"}')

# sending get request and saving the response as response object 
results = requests.post(url = URL, json = DATA)
print(results)

# extracting data in json format 
data = results.json()

# extracting the subject and body data 
subjects = []
bodies = []
for d in data['value']:
    subjects.append(d['subject'])
    bodies.append(d['body'])

# printing the output 
i = 0
while i < len(subjects):
    print("Subject:%s\nBody:%s\n"%(subjects[i], bodies[i])) 
    i += 1