from itertools import combinations_with_replacement

class Item:
    def __init__(self, stored, value: float = 0.0, size: int = 0):
        self.value = value
        self.size = size
        self.stored = stored

def np_knapsack(
    items_a: list[Item],
    items_b: list[Item],
    C: int,
    min_items: int = 1,
    max_items: int = 5,
    lambda_val: float = 0.5
) -> list[Item]:
    """
    Knapsack selection allowing duplicates from items_b.
    Fills remaining capacity dynamically with small items.
    Adds a preference for fewer items.
    """

    if not items_b or C <= 0:
        return []

    # Filter items that can fit at all
    fit_items = [item for item in items_b if item.size <= C]
    if not fit_items:
        return []

    a_values = [a.value for a in items_a]
    best_subset: list[Item] = []
    best_score = -float('inf')

    ALPHA = lambda_val
    BETA = 1 - lambda_val
    GAMMA = 0.01  # small penalty for more items

    max_possible_items = min(max_items, C // min(item.size for item in fit_items))
    min_possible_items = max(min_items, 1)

    for k in range(min_possible_items, max_possible_items + 1):
        for combo in combinations_with_replacement(fit_items, k):
            total_size = sum(item.size for item in combo)
            if total_size > C:
                continue

            # Fill remaining capacity with small items
            remaining = C - total_size
            filler = []
            if remaining > 0:
                small_items = sorted(fit_items, key=lambda x: x.size)
                while remaining > 0:
                    for item in small_items:
                        if item.size <= remaining:
                            filler.append(item)
                            remaining -= item.size
                            break
                    else:
                        break

            full_combo = list(combo) + filler
            final_total_size = sum(item.size for item in full_combo)
            fill_ratio = final_total_size / C

            if a_values:
                dist_sum = sum(min(abs(item.value - a) for a in a_values) for item in full_combo)
                avg_distance = dist_sum / len(full_combo)
                similarity_score = 1.0 / (1.0 + avg_distance)
            else:
                similarity_score = 0.0

            # Add penalty for more items
            score = ALPHA * fill_ratio + BETA * similarity_score - GAMMA * len(full_combo)

            # Select best score, prefer fewer items if equal
            if score > best_score or (score == best_score and len(full_combo) < len(best_subset)):
                best_score = score
                best_subset = full_combo

    if not best_subset:
        # fallback: pick best single item
        best_subset = [max(fit_items, key=lambda x: x.value / x.size)]

    return best_subset

def test():
    items_a: list[Item] = [
        Item('~', 17, 13),
        Item('&', 8, 15),
        Item('=', 7, 12),
        Item('^', 11, 11),
        Item('#', 15, 14),
        Item('*', 9, 9),
        Item('!', 8, 5),
        Item('M', 9, 18),
        Item('_', 9, 15),
        Item(';', 6, 5),
        Item('-', 7, 10),
        Item('$', 13, 13),
        Item('.', 3, 5),
        Item('+', 5, 12),
        Item(':', 4, 5),
        Item('i', 10, 5),
        Item('[', 13, 8),
        Item('%', 6, 19),
        Item('\'', 13, 5),
        Item('l', 11, 5),
        Item('B', 12, 14),
        Item(' ', -10000, 6),
        Item('@', 11, 23),
        Item('8', 10, 13),
        Item('W', 7, 23),
        Item(',', 5, 5),
        Item(']', 15, 8)
    ]

    items_b: list[Item] = [
        Item('\'', 13, 5),
        Item('filler', 1, 1)
    ]

    C = 12
    knapsack = np_knapsack(items_a, items_b, C)

    for item in knapsack:
        print(item.stored, item.value, item.size)

if __name__ == '__main__':
    test()
