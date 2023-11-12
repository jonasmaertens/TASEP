import json
import os

with open("models/all_models.json", "r") as f:
    all_models = json.load(f)
for model_id in all_models:
    print(model_id)
    all_models[model_id]["random_density"] = True

with open("models/all_models.json", "w") as f:
    json.dump(all_models, f)
