from models.rl.supervised import SupervisedActionModel


def main() -> None:
    model = SupervisedActionModel.load("checkpoints/rl_agent_supervised.joblib")
    calibration = model.metadata.get("post_cost_calibration", {})
    for side in ("long", "short"):
        data = calibration.get(side, {})
        print("===", side, "global", data.get("global"))
        rows = data.get("thresholds", [])
        for row in rows[:12]:
            print(row)


if __name__ == "__main__":
    main()
