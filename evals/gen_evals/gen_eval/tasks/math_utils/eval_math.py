from collections import Counter
from .math_grader import extract_answer, math_equal

class MathMetrics:
    def __init__(self):
        self.reset()

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        self.total += 1

        current_correct = False

        if aggregation_mode == "best":
            current_correct = any([elem['is_correct'] for elem in predictions])
            if all([elem['predicted_answer'] is None for elem in predictions]):
                self.no_answer += 1
        elif aggregation_mode == "majority":
            valid_answers_and_results = [
                (elem['predicted_answer'], elem['is_correct'])
                for elem in predictions
                if elem['predicted_answer'] is not None
            ]
            if len(valid_answers_and_results) == 0:
                self.no_answer += 1
            else:
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                current_correct = majority_result[1]
        elif aggregation_mode == "first":
            current_correct += predictions[0]['is_correct']
            self.no_answer += predictions[0]['predicted_answer'] is None
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

        self.correct += current_correct

    def get_metrics(self):
        metrics = {"num_entries": self.total}
        metrics["symbolic_correct"] = self.correct / self.total * 100.0
        metrics["no_answer"] = self.no_answer / self.total * 100.0
        return metrics

    def reset(self):
        self.correct = 0
        self.no_answer = 0
        self.total = 0

def compute_metrics(
    all_predictions,
    aggregation_mode='first',
):
    metrics_calculator = MathMetrics()
    for predictions in all_predictions:
        metrics_calculator.update(predictions, aggregation_mode)
    return metrics_calculator.get_metrics()

def _eval_single_example(
    generation,
    expected_answer,
    extract_from_boxed=True,
    extract_regex=r"The final answer is (.+)$",
    include_percentage=True,
    tolerance=1e-4,
    timeout=10.0,
):
    predicted_answer = extract_answer(
        generation,
        extract_from_boxed=extract_from_boxed,
        extract_regex=extract_regex,
    )
    try:
        output = math_equal(
            predicted_answer,
            expected_answer,
            include_percentage,
            tolerance,
            timeout,
        )
        error_message = ""
    except Exception as e:
        output = False
        error_message = str(e)
        # print(e, flush=True)

    return predicted_answer, output, error_message

def eval_all_outputs(
    generations,
    references,
    extract_from_boxed=True,
    extract_regex=r"The final answer is (.+)$",
    include_percentage=True,
    tolerance=1e-4,
    timeout=10.0,
):
    outputs = {}
    all_predictions = []
    for task_id, (gens, ref) in enumerate(zip(generations, references)):
        predictions = []
        for gen in gens:
            predicted_answer, is_correct, err_msg = _eval_single_example(
                gen, 
                ref,
                extract_from_boxed=extract_from_boxed,
                extract_regex=extract_regex,
                include_percentage=include_percentage,
                tolerance=tolerance,
                timeout=timeout,
            )
            predictions.append(
                {
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                }
            )
        all_predictions.append(predictions)
    
    outputs["greedy"] = compute_metrics(
        all_predictions,
        aggregation_mode="first",
    )
    num_samples = len(all_predictions[0])
    if num_samples > 1:
        outputs[f'majority@{num_samples}'] = compute_metrics(
            all_predictions,
            aggregation_mode="majority",
        )
        outputs[f'best@{num_samples}'] = compute_metrics(
            all_predictions,
            aggregation_mode="best",
        )
    return outputs
