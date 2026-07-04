"""Modal serverless endpoint: POST a player's raw stats, get back a draft probability."""
import modal

# A named handle Modal groups this deployment's function(s) and public URL under.
app = modal.App("nfl-draft-prediction")

# The container recipe: pinned serving stack + FastAPI + our package + the model artifacts.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lightgbm==4.6.0",
        "catboost==1.2.10",
        "scikit-learn==1.9.0",
        "pandas==3.0.3",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "joblib==1.5.3",
        "shap==0.51.0",
        "matplotlib==3.10.9",
        "fastapi[standard]",
    )
    .env({"NFL_DRAFT_MODELS_DIR": "/models"})   # tell config.py where the artifacts live
    .add_local_dir("models", "/models")          # copy lgbm/catboost/preprocess into the image
    .add_local_python_source("nfl_draft")        # make `import nfl_draft` work in the container
)


# A public POST URL that runs predict_one on the incoming player JSON in the cloud.
@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def predict(player: dict):
    from nfl_draft.models.predict import predict_one   # heavy imports run in the container, not locally
    return predict_one(player)
