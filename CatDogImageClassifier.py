import random

def player(prev_p.append(prev_play)
    my_history = player.my_history

    # Define counters for each move
    counter_moves = {"R": "P", "P": "S", "S": "R"}

    # Helper function to get most frequent move in a history slice
    def most_frequent(history, n=10):
        if not history:
            return random.choice(["R", "P", "S"])
        recent = history[-n:] if len(history) >= n else history
        counts = {"R": 0, "P": 0, "S": 0}
        for move in recent:
            counts[move] += 1
        max_count = max(counts.values())
        most_common = [move for move, count in counts.items() if count == max_count]
        return random.choice(most_common)

    # Default move for the first game or if no pattern is detected
    guess = random.choice(["R", "P", "S"])

    # Identify opponent based on move patterns
    if len(opponent_history) >= 5:
        # Quincy: Check for repeating sequence R, P, P, S, R
        quincy_sequence = ["R", "P", "P", "S", "R"]
        is_quincy = True
        for i in range(min(len(opponent_history), 5)):
            if opponent_history[-i-1] != quincy_sequence[-(i % 5 + 1)]:
                is_quincy = False
                break
        if is_quincy:
            next_quincy = quincy_sequence[len(opponent_history) % 5]
            guess = counter_moves[next_quincy]

        # Kris: Plays counter to our last move
        elif prev_play == counter_moves.get(my_history[-1] if my_history else ""):
            # If we played X last, Kris plays counter(X), so we play counter(counter(X))
            last_move = my_history[-1] if my_history else random.choice(["R", "P", "S"])
            kris_counter = counter_moves[last_move]
            guess = counter_moves[kris_counter]

        # Mrugesh: Plays counter to our most frequent move in last 10
        elif len(my_history) >= 10:
            our_most_frequent = most_frequent(my_history, 10)
            mrugesh_counter = counter_moves[our_most_frequent]
            # Play counter to Mrugesh's counter
            guess = counter_moves[mrugesh_counter]

        # Abbey: Tries to counter our patterns, so we mix strategies
        else:
            # Play counter to our most frequent move to confuse Abbey
            our_most_frequent = most_frequent(my_history)
            guess = counter_moves[our_most_frequent]

    # Store our move
    my_history.append(guess)

    return guess