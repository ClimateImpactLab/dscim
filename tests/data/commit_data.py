"""
Commit data from Gitlab CI/CD runner to repo
"""
import os
import requests
from urllib.parse import quote, urljoin


def post_commit_data():
    """API REST Put to repo"""
    api_endpoint = "https://gitlab.com/api/v4/projects/"
    token = os.environ.get("GITLAB_PERSONAL_TOKEN", "")
    file_obj = open("dscim/tests/data/menu_results.zip", "rb")
    data_dict = {
        "branch": "main",
        "author_email": os.environ.get("EMAIL"),
        "author_name": os.environ.get("NAME"),
        "content": "Test data",
        "commit_message": ":floppy_disk: Bump test data",
    }
    files = {"file": file_obj}

    headers = {"PRIVATE-TOKEN": token}
    file_path = quote("dscim/tests/data/menu_results.zip", safe="")
    params = f"20123675/repository/files/{file_path}"
    full_url = urljoin(api_endpoint, params)

    response = requests.post(full_url, data=data_dict, headers=headers, files=files)

    if response.status_code == requests.codes.ok:
        print(":robot: Success! Commit done!")
    else:
        print(f"Oops! Something went wrong: {response.json()}")

    return None


if __name__ == "__main__":
    post_commit_data()
