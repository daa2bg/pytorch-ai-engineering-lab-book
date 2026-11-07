"""Reference answer for a toy eligibility trace; no biological equivalence claimed."""


def eligibility_updates(events, *, decay=0.5, learning_rate=0.1):
    if not 0 <= decay <= 1 or learning_rate < 0:
        raise ValueError("decay must be in [0,1] and learning rate nonnegative")
    trace = 0.0
    weight = 0.0
    history = []
    for activity, modulator in events:
        trace = decay * trace + activity
        weight += learning_rate * modulator * trace
        history.append(weight)
    return history
