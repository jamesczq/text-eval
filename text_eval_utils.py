"""
Contains funtionality to evaluate machine answers against true answers.

James 03/21/2023
"""

from typing import List, Dict
from rouge import Rouge


class PrecisionRecallF1(Rouge):
    def __init__(self, machine_answer, true_answer):
        super().__init__()
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self._set_scores(machine_answer, true_answer)

    def _set_scores(self, machine_answer: str, true_answer: str):
        scores = self.get_scores(machine_answer, true_answer)[0]["rouge-1"]
        self.precision = scores["p"]
        self.recall = scores["r"]
        self.f1 = scores["f"]


def get_batch_metrics(
    array_machine_answers: List[str], 
    array_true_answers: List[str]
) -> Dict[str, List[float]]:

    assert len(array_machine_answers) == len(array_true_answers)

    N = len(array_machine_answers)

    precisions = [0.0] * N
    recalls = [0.0] * N
    f1s = [0.0] * N

    for i in range(N):
        try:
            scores = PrecisionRecallF1(
                array_machine_answers[i], 
                array_true_answers[i]
            )

            precisions[i] = scores.precision
            recalls[i] = scores.recall
            f1s[i] = scores.f1

        except:
            pass

    dict_out = {"precision": precisions, "recall": recalls, "f1": f1s}

    return dict_out
