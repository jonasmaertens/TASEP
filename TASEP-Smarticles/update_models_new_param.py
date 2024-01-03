import json
import os

with open("models/all_models.json", "r") as f:
    all_models = json.load(f)
for model_id in all_models:
    print(model_id)
    if "new_model" not in all_models[model_id]:
        all_models[model_id]["new_model"] = False

with open("models/all_models.json", "w") as f:
    json.dump(all_models, f, indent=4)
