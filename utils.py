import json

def read_settings(filename="settings.json"):
    with open(filename, 'r') as file:
        return json.load(file)