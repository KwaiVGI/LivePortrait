mapping = {
        "src_key": "../assets/examples/source/s5.jpg",
        "driving_key": "../assets/examples/driving/d13.mp4",
    }

import requests, json

base_url = "http://localhost:8005"
headers = {"accept": "application/json", "Content-Type": "application/json"}

response = requests.post(f"{base_url}/submit", headers=headers, json=mapping)
print(response)
