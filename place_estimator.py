# place_estimator.py

from weights import PLACE_WEIGHTS


def estimate_place(detected_counts):

    scores = {}

    for place, weights in PLACE_WEIGHTS.items():

        score = sum(
            weights.get(obj_name, 0) * count
            for obj_name, count in detected_counts.items()
        )

        scores[place] = score

    max_score = max(scores.values())

    if max_score == 0:
        return "unknown", scores

    candidates = [
        place
        for place, score in scores.items()
        if score == max_score
    ]

    best_place = candidates[0]

    return best_place, scores