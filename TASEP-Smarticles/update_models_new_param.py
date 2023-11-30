import json
import os

with open("models/all_models.json", "r") as f:
    all_models = json.load(f)
for model_id in all_models:
    print(model_id)
    all_models[model_id]["num_models"] = 1

    # with open(os.path.join(all_models[model_id]["path"], "env_params.json"), "w") as f:
    #     json.dump(all_models[model_id]["env_params"], f)

with open("models/all_models.json", "w") as f:
    json.dump(all_models, f, indent=4)
