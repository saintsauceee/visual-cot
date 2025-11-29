import json
import os
print("CWD:", os.getcwd())
errors_file = "./eval/results/gpt/few_shots/batch_errors.jsonl"          # file with responses/errors
requests_file = "./eval/results/gpt/few_shots/requests_1.jsonl"      # file with original requests
output_file = "./eval/results/gpt/few_shots/requests_2.jsonl" # file to write failed requests

# -----------------------------------------------------
# 1. Collect custom_ids of failed requests
# -----------------------------------------------------
failed_ids = set()

with open(errors_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        resp = obj.get("response", {})
        code = resp.get("status_code")

        if code != 200:
            failed_ids.add(str(obj.get("custom_id")))

print(f"Found {len(failed_ids)} failed requests.")


# -----------------------------------------------------
# 2. Separate failed vs successful requests
# -----------------------------------------------------
failed_requests = []
successful_requests = []

with open(requests_file, "r") as req_f:
    for line in req_f:
        if not line.strip():
            continue
        req_obj = json.loads(line)

        if str(req_obj.get("custom_id")) in failed_ids:
            failed_requests.append(req_obj)
        else:
            successful_requests.append(req_obj)

print(f"Moving {len(failed_requests)} failed requests.")
print(f"Keeping {len(successful_requests)} successful requests.")


# -----------------------------------------------------
# 3. Write failed ones to failed_requests.jsonl
# -----------------------------------------------------
with open(output_file, "w") as out_f:
    for obj in failed_requests:
        out_f.write(json.dumps(obj) + "\n")


# -----------------------------------------------------
# 4. Overwrite requests.jsonl with ONLY successful ones
# -----------------------------------------------------
with open(requests_file, "w") as req_f:
    for obj in successful_requests:
        req_f.write(json.dumps(obj) + "\n")

print("Done. Failed requests moved and removed from the original file.")
