# Initialize variables

score = 0
high_score = 0


def check_score():
    return score


def increase_score():
    global score
    score += 1


def reset_score():
    global score
    score = 0


def check_high_score():
    return high_score


def set_high_score(i):
    global high_score
    high_score = i
