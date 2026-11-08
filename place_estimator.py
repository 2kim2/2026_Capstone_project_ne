# place_estimator.py
from weights import PLACE_WEIGHTS


def estimate_place(objects):

    scores = {}

    for place, weights in PLACE_WEIGHTS.items():

        score = 0

        for obj in objects:

            obj_name = obj["name"]
            confidence = obj["confidence"]

            weight = weights.get(obj_name, 0)

            score += weight * confidence

        scores[place] = round(score, 4)

    max_score = max(scores.values())

    if max_score == 0:
        return "인식할 수 있는 사물이 없습니다.", scores

    candidates = [
        place
        for place, score in scores.items()
        if score == max_score
    ]

    best_place = candidates[0]

    return best_place, scores