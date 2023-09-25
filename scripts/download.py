import os
from huggingface_hub import hf_hub_download

with open(os.path.join(os.path.dirname(__file__), "download_list.txt"), "r") as r:
    download_list = r.read().splitlines()

for download in download_list:
    repo_id = "stabilityai/" + download.split("/")[1]
    subfolder = download.split("/")[2]
    filename = download.split("/")[-1]

    local_dir = download.split("/")[0] + "/" + download.split("/")[1]
    local_dir = os.path.join(os.path.dirname(__file__), "..", local_dir)

    hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=local_dir, local_dir_use_symlinks=False)