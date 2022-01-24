import json
def celeb_names():
    names_file = open(r'C:\Users\HP\Machine Learning Course\Celebrity Face Recognition\celeb_names.json')
    names = json.load(names_file)
    return names