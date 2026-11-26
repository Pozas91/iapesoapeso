"""Mean, variance and the bell curve: the numbers of route video 07 (folder 35).

Two bus lines with the same mean and a different spread; why the deviations are squared (the
mean is the constant with the smallest mean squared error, video 02); the standard deviation;
how far a value is in standard deviations; coins and dice that add up to a bell; a coin game that
multiplies instead of adding and ends in a long tail; variances that add up;
and a neuron that sums 100 products, whose output spreads ten times more than its inputs.
Run:  python bell.py
"""

import math
import random

BUS_A = [17, 19, 20, 21, 23]
BUS_B = [13, 19, 21, 22, 25]


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    """Mean of the squared distances to the mean."""
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def mse_constant(xs, c):
    """Mean squared error of always predicting c (video 02)."""
    return sum((x - c) ** 2 for x in xs) / len(xs)


def heads_distribution(n):
    """Probability of each number of heads with n fair coins."""
    return [math.comb(n, k) / 2**n for k in range(n + 1)]


def dice_distribution(n):
    """Probability of each total with n fair dice (index = total), adding one die at a time."""
    dist = [1.0]
    for _ in range(n):
        new = [0.0] * (len(dist) + 6)
        for total, p in enumerate(dist):
            for face in range(1, 7):
                new[total + face] += p / 6
        dist = new
    return dist


def coin_game(rounds=20, start=100, heads=1.5, tails=0.6):
    """Money after each number of heads when every head multiplies it by `heads` and every tail by
    `tails`, with the probability of that number of heads."""
    return [(math.comb(rounds, k) / 2**rounds, start * heads**k * tails ** (rounds - k)) for k in range(rounds + 1)]


def normal_cdf(z):
    """Share of the bell (mean 0, standard deviation 1) below z."""
    return (1 + math.erf(z / math.sqrt(2))) / 2


def neuron_outputs(w_std, inputs=100, trials=10_000, seed=7):
    """Outputs of a neuron that sums `inputs` products of an input and a weight, both drawn from bells
    of mean 0 (inputs with std 1, weights with std `w_std`), over `trials` draws."""
    rng = random.Random(seed)
    return [sum(rng.gauss(0, 1) * rng.gauss(0, w_std) for _ in range(inputs)) for _ in range(trials)]


def gpt2_small_weights(vocab=50257, context=1024, width=768, layers=12):
    """Parameters of GPT-2 small (OpenAI's hparams), all drawn from a bell before training."""
    block = (
        2 * width  # first layer norm
        + width * 3 * width
        + 3 * width  # attention: queries, keys and values
        + width * width
        + width  # attention output
        + 2 * width  # second layer norm
        + width * 4 * width
        + 4 * width  # feed-forward up
        + 4 * width * width
        + width  # feed-forward down
    )
    return vocab * width + context * width + layers * block + 2 * width


def main():
    print(f"0. GPT-2 small: {gpt2_small_weights():,} parameters; weights drawn with std 0.02")
    print("1. Two buses, five days each")
    for name, xs in (("A", BUS_A), ("B", BUS_B)):
        m = mean(xs)
        deviations = [x - m for x in xs]
        print(f"   {name}: {xs}  mean {m:g}  deviations {[f'{d:+g}' for d in deviations]}  sum {sum(deviations):g}")
        squares = [d * d for d in deviations]
        print(f"      squared {[f'{s:g}' for s in squares]}  sum {sum(squares):g}  variance {variance(xs):g}")
        print(f"      absolute mean {mean([abs(d) for d in deviations]):g}  std {math.sqrt(variance(xs)):g}")
        print(f"      mean + 2 std = {m + 2 * math.sqrt(variance(xs)):g} min")

    print("2. The mean is the constant with the smallest mean squared error (bus A)")
    for c in (18, 19, 20, 21, 22):
        print(f"   always predict {c}: MSE {mse_constant(BUS_A, c):g}")

    print("3. How far, in standard deviations (z = (x - mean) / std)")
    for name, xs in (("A", BUS_A), ("B", BUS_B)):
        m, s = mean(xs), math.sqrt(variance(xs))
        print(f"   a 26-minute day on {name}: (26 - {m:g}) / {s:g} = {(26 - m) / s:g}")
    print(f"   bus B normalised: {[f'{(x - mean(BUS_B)) / math.sqrt(variance(BUS_B)):+g}' for x in BUS_B]}")

    print("4. Coins: number of heads")
    for n in (1, 2, 4, 10, 100):
        dist = heads_distribution(n)
        m = sum(k * p for k, p in enumerate(dist))
        v = sum((k - m) ** 2 * p for k, p in enumerate(dist))
        print(f"   {n:3} coins: mean {m:g}  variance {v:g}  std {math.sqrt(v):g}")
    print(f"   2 coins: {[f'{p:g}' for p in heads_distribution(2)]}")
    print(f"   4 coins: {[f'{p:g}' for p in heads_distribution(4)]}")
    d100 = heads_distribution(100)
    print(f"   100 coins: exactly 50 heads {d100[50]:.4f}")
    print(f"   100 coins: 45 to 55 heads {sum(d100[45:56]):.4f}  40 to 60 {sum(d100[40:61]):.4f}")
    print(f"   400 coins: std {math.sqrt(400 * 0.25):g}")
    print(f"   100 coins: ways to get 50 heads {math.comb(100, 50):.3g}, 90 heads {math.comb(100, 90):.3g}")

    print("4b. Dice: any shape adds up to a bell")
    print(f"   one die: mean {mean(range(1, 7)):g}  {[f'{p:.3f}' for p in dice_distribution(1)[1:]]}")
    two = dice_distribution(2)
    print(f"   two dice: ways {[round(p * 36) for p in two[2:]]} of 36 (totals 2 to 12)")
    ten = dice_distribution(10)
    m = sum(t * p for t, p in enumerate(ten))
    print(f"   ten dice: mean {m:g}  most likely total {ten.index(max(ten))}  P(35) {ten[35]:.4f}")
    print(f"   ten dice: 25 to 45 {sum(ten[25:46]):.4f}  10 or 60 {ten[10]:.2g} each")

    print("4c. Adding or multiplying: 20 coins, 100 euros")
    print("   adding: heads +10, tails -10 ->", [100 + 10 * (2 * k - 20) for k in (0, 5, 10, 15, 20)])
    game = coin_game()
    print(f"   multiplying: one head and one tail: 100 * 1.5 = {100 * 1.5:g}, * 0.6 = {100 * 1.5 * 0.6:g}")
    print(f"   a head and a tail multiply by 1.5 * 0.6 = {1.5 * 0.6:g}: less than 1, whatever the order")
    print(
        f"   average factor per round: (1.5 + 0.6) / 2 = {(1.5 + 0.6) / 2:g}; after 20: 100 * 1.05^20 = {100 * 1.05**20:.2f}"
    )
    print(f"   mean of the game: {sum(p * v for p, v in game):.2f}")
    for k in (5, 10, 11, 12, 15, 17, 20):
        print(f"   {k:2} heads: {game[k][1]:,.2f} euros  (probability {game[k][0]:.5f})")
    print(f"   lose money (less than 100): {sum(p for p, v in game if v < 100):.4f}")
    print(f"   more than 20,000 euros: {sum(p for p, v in game if v > 20_000):.5f}")

    print("5. The bell (mean 0, std 1)")
    for k in (1, 2, 3):
        print(f"   within {k} std: {normal_cdf(k) - normal_cdf(-k):.4f}")
    print(f"   below 1 std: {normal_cdf(1):.4f}   (GELU of 1, video 04: {1 * normal_cdf(1):.3f})")

    print("6. A neuron that sums 100 inputs times weights, all with mean 0 and std 1")
    for w_std in (1.0, 0.1):
        sums = neuron_outputs(w_std)
        print(
            f"   weights std {w_std:g}: output std {math.sqrt(variance(sums)):.2f} (theory {math.sqrt(100) * w_std:g})"
        )
    print(f"   ten such layers in a row, weights std 1: std 10^10 = {10**10:,}")


if __name__ == "__main__":
    main()
