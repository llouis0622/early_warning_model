from huggingface_hub import HfApi

# 설정
username = "LLouis0622"
repo_name = "early_warning_model"
repo_id = f"{username}/{repo_name}"

print(f"📦 Uploading to: https://huggingface.co/{repo_id}")

api = HfApi()

print("Uploading files (this may take a while)...")
try:
    api.upload_large_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "data/raw/*",
            "*.zip",
            ".git/*",
            ".git",
            "__pycache__/*",
            "*.pyc",
            ".gitignore",
            ".venv/*",
            "venv/*",
            ".DS_Store",
            ".idea/*",
            "*.egg-info/*",
            ".ipynb_checkpoints/*",
            "early_warning_model/*"
        ],
        multi_commits=True,
        multi_commits_verbose=True
    )
    print(f"Upload complete!")
    print(f"View at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Error: {e}")
    print("Try uploading in smaller batches or excluding large files")