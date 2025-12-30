from itertools import combinations_with_replacement

class Item:
    def __init__(self, stored, value: float = 0.0, size: int = 0):
        self.value = value
        self.size = size
        self.stored = stored

def np_knapsack(items_a: list[Item],
                items_b: list[Item],
                C: int,
                min_items: int = 3,
                max_items: int = 5,
                lambda_val: float = 0.5) -> list[Item]:
    """
    * ChatGPT-5
    Selects items from items_b (allowing duplicates) to maximize capacity usage
    while keeping values similar to items_a.

    Args:
        items_a: list of reference Items (values only)
        items_b: list of candidate Items (size and value)
        C: capacity limit (sum of sizes)
        min_items: minimum expected number of items
        max_items: maximum expected number of items
        lambda_val:
    Returns:
        best_subset: list of selected Items from B (may contain duplicates)
    """

    if not items_b or C <= 0:
        return []

    a_values = [a.value for a in items_a]
    best_subset: list[Item] = []
    best_score = -float('inf')

    ALPHA = lambda_val  # weight for capacity fill
    BETA = 1 - lambda_val   # weight for value similarity

    for k in range(min_items, max_items + 1):
        # combinations with replacement allow duplicates
        for combo in combinations_with_replacement(items_b, k):
            total_size = sum(item.size for item in combo)
            if total_size > C:
                continue  # skip if capacity exceeded

            # ---- fill score ----
            fill_ratio = total_size / C

            # ---- similarity score ----
            if a_values:
                dist_sum = 0.0
                for item in combo:
                    nearest = min(abs(item.value - a) for a in a_values)
                    dist_sum += nearest
                avg_distance = dist_sum / k
                similarity_score = 1.0 / (1.0 + avg_distance)
            else:
                similarity_score = 0.0

            # ---- combined score ----
            score = ALPHA * fill_ratio + BETA * similarity_score

            if score > best_score:
                best_score = score
                best_subset = list(combo)

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
        Item(' ', -10000, 6),
        Item('.', 3, 5),
        Item(':', 4, 5),
        Item('+', 5, 12),
    ]

    knapsack = np_knapsack(items_a, items_b, 25)
    for item in knapsack:
        print(item.stored, item.value, item.size)

if __name__ == '__main__':
    test()
