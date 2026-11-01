"""Solutions of exercises.md. Run:  python solutions.py"""

from faithful_case import DURATION, exact, steps_to_arrive

m, n = exact(DURATION)
print("== 1. The ranger's forecast ==")
for minutes in (2, 4.5):
    print(f"after a {minutes}-minute eruption: {m * minutes + n:.1f} minutes")


def largest_stable_eta(xs, low, high):
    """Bisection between an eta that arrives (low) and one that explodes (high)."""
    for _ in range(30):
        mid = (low + high) / 2
        low, high = (mid, high) if steps_to_arrive(mid, xs, max_steps=20_000) else (low, mid)
    return low


mean = sum(DURATION) / len(DURATION)
centred = [x - mean for x in DURATION]
print("== 2. The largest eta that does not explode (approximately) ==")
print(f"raw: {largest_stable_eta(DURATION, 0.01, 0.1):.3f}")
print(f"centred: {largest_stable_eta(centred, 0.1, 1.0):.3f}")

print("== 3. What n means ==")
print(f"raw: n = {n:.1f}, the wait after an eruption of 0 minutes")
print(f"centred: n = {exact(centred)[1]:.1f}, the wait after an eruption of average length ({mean:.2f} min)")
