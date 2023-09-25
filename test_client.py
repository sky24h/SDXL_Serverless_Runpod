import runpod
import base64

def decode_data(data, save_path):
    fh = open(save_path, "wb")
    fh.write(base64.b64decode(data))
    fh.close()

# Set your API key here
runpod.api_key = ""

# Set your endpoint ID here
endpoint = runpod.Endpoint("")
print("Waiting for response...")
run_request = endpoint.run_sync(
    {
        "prompt": "A beautiful white cat, looking at the viewer, sitting next to a window, sun shining through the window.",
        "steps": 40,
        "width": 960,
        "height": 1280,
    }
)
result = run_request
if result["status"] == "COMPLETED":
    print("Generation completed successfully!")
    print("Here's your image:")
    filename = result["output"]["filename"]
    save_path = "/tmp/" + filename
    data = result["output"]["data"]
    decode_data(data, save_path)
    print("Saved to " + save_path)
else:
    print("Generation failed :(")
    print(result)