# 04 · Jenkins CI pipeline

A Jenkins job for the used-car price model. It downloads and cleans the data (`download.py`), trains with MLflow logging (`train_model.py`), serves the best model with `mlflow models serve` on port 5003 and sends a test request to it. The build steps are in `jenkins_pipline.sh`.

Run it locally without Jenkins:

```bash
pip install -r requirements.txt
python download.py && python train_model.py > best_model.txt
mlflow models serve -m "$(tail -1 best_model.txt)" -p 5003 --env-manager local
```
