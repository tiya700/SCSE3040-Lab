"Predict one delivery, from the command line."

from pathlib import Path

import pandas as pd

from delivery import load_model


HERE = Path(__file__).resolve().parent


def main():
    model = load_model(HERE / "model.joblib")
    order = pd.DataFrame([{
        "distance_km": 7.0,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0,
    }])
    minutes = float(model.predict(order)[0])
    print(f"PREDICTION: {minutes:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
