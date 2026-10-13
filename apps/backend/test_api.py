import requests

url = "http://127.0.0.1:8000/api/v1/Get_N_moreJDs_for_Res"
payload = {
    "N": 10,
    "LastJDid": "123",
    "ResumeID": "c94b738a1558c9db558d2037fb455ec691f6f4de19d66bbc9ea1135b0ae893d3",
    "top_k": 1000
}
try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
