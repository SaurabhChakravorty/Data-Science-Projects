import requests

base_url = "http://127.0.0.1:8080"

# Check server readiness
ready = requests.get(f"{base_url}/ready")
if ready.status_code != 200:
    print("**Server is not ready**", ready.text)
    exit()
print("--Server is ready--")

# Define test inputs
test_texts = [
    "Book me a flight to New York",
    "I'd like to find hotels near Boston",
    "What's the weather in San Francisco?",
    "hello",
    "comeon baby",  # noisy input
    "",
]

# Run predictions
for text in test_texts:
    print(f"\n Input: {text}")
    response = requests.post(f"{base_url}/intent", json={"text": text})
    try:
        print("Output:", response.json())
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", response.text)
