## To create requirement.txt automatically
pip3 freeze > requirements.txt   

## Install Packages from requirements.txt:
pip3 install -r requirements.txt

## Verify Installation
pip3 list

## Deploy to Render Steps
# Step 1: Create a virtual environment
python3 -m venv venv
# Step 2: Activate it
source venv/bin/activate

