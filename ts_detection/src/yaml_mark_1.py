import yaml

# Load the YAML file into a Python dictionary
with open("../iqa.yaml") as file:
    data = yaml.safe_load(file)

# Access the values in the dictionary
print(data[10]["mre"]["sigma"]) # Output: John Doe
#print(data["person2"]["address"]["zip"]) # Output: 94103