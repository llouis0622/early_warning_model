from huggingface_hub import HfApi, create_repo

username = "LLouis0622"
repo_name = "early_warning_model"
repo_id = f"{username}/{repo_name}"

api = HfApi()

try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Repository already exists or error: {e}")

api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=[
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        ".gitignore",
        "venv/*",
        ".DS_Store"
    ]
)

print(f"Upload complete! Check: https://huggingface.co/{repo_id}")

from huggingface_hub import HfApi, create_repo

username = "LLouis0622"
repo_name = "early_warning_model"
repo_id = f"{username}/{repo_name}"

api = HfApi()

try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Repository already exists or error: {e}")

api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=[
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        ".gitignore",
        "venv/*",
        ".DS_Store"
    ]
)

print(f"Upload complete! Check: https://huggingface.co/{repo_id}")


