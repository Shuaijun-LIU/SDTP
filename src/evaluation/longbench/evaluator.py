import json
import os
from typing import Dict, List, Optional
from .dataset import LongBenchDataset
from .model_wrapper import ModelWrapper

def _simple_match(pred: str, answers: List[str]) -> float:
    pred = pred.lower()
    for ans in answers:
        if ans.lower() in pred:
            return 1.0
    return 0.0

class LongBenchEvaluator:
    def __init__(self, task_path: str, model: ModelWrapper, max_samples=None):
        self.task_path = task_path
        self.dataset = LongBenchDataset(task_path)
        self.model = model
        self.max_samples = max_samples

    def evaluate(
        self,
        do_inference: bool = False,
        save_predictions: bool = False,
        prediction_path: Optional[str] = None,
    ) -> Dict:
        total = len(self.dataset)
        if self.max_samples:
            total = min(total, self.max_samples)

        print(f"[Eval] Task loaded: {len(self.dataset)} total")
        print(f"[Eval] Evaluating: {total} samples")
        print(f"[Eval] Model: {self.model.model_name}")

        if not do_inference:
            print("[Eval] Setup only. No inference.")
            return {
                "task": self.task_path,
                "num_total": len(self.dataset),
                "num_eval": total,
                "model": self.model.model_name,
                "mode": self.model.mode,
                "status": "setup_completed",
            }

        print("[Eval] >>> Real inference START <<<")

        hits = 0
        preds = []

        for i in range(total):
            item = self.dataset[i]
            pred = self.model.infer(item["input"])
            hit = _simple_match(pred, item["answers"])
            hits += hit

            preds.append({
                "id": i,
                "input": item["input"],
                "prediction": pred,
                "answers": item["answers"],
                "hit": hit,
            })

        hit_rate = hits / total

        print("[Eval] DONE")
        print(f"[Eval] Hit Rate: {hit_rate:.4f}")

        if save_predictions:
            if prediction_path is None:
                base = os.path.basename(self.task_path).replace(".json", "")
                prediction_path = f"results/longbench_{base}_pred.json"

            os.makedirs("results", exist_ok=True)
            with open(prediction_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)

            print(f"[Eval] Predictions saved to {prediction_path}")

        return {
            "task": self.task_path,
            "num_eval": total,
            "hit_rate": hit_rate,
            "model": self.model.model_name,
            "status": "inference_completed",
        }
