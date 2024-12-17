import modal
from pp import get_projections
import os

# Create a stub for the Modal app
stub = modal.Stub("prizepicks-scraper")

# Create an image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    # Copy all necessary files into the container
    .copy_local_file(".env", "/root/.env")
    .copy_local_file("pp.py", "/root/pp.py")
    .copy_local_file("dump_to_db.py", "/root/dump_to_db.py")
    .copy_local_dir("db", "/root/db")  # Copy the entire db directory
)

@stub.function(
    image=image,
    schedule=modal.Period(minutes=30), timeout=900
)
def run_scraper():
    print("Starting PrizePicks scraper...")
    # Add the root directory to Python path so it can find the modules
    import sys
    sys.path.append("/root")
    
    get_projections()
    print("Scraping completed")

@stub.local_entrypoint()
def main():
    run_scraper.remote()
