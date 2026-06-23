import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from models.ml_engine import MatchOutcomeModel

log = get_logger("train")

def main():
    log.info("Building training dataset with latest production logic...")
    pipeline = MatchDataPipeline()
    from data.feature_engineering import build_training_features
    X, y = build_training_features(pipeline.training_df, None)
    
    log.info(f"Training dataset ready: {X.shape[0]} rows, {X.shape[1]} features.")
    
    model = MatchOutcomeModel()
    model.fit(X, y)
    model.save()
    
    log.info("Production model retrained and saved successfully.")

if __name__ == "__main__":
    main()
