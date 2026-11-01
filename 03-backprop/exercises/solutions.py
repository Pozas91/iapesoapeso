"""Solutions of exercises.md. Run:  python solutions.py  (takes about a minute)"""

from iris_net_case import ALL, PETAL, Y, errors, initial, standardised, train, twins

print("== 1. More hidden neurons, petal only ==")
xs = standardised(PETAL)
for hidden in (2, 4, 8):
    print(f"{hidden} hidden neurons: {errors(train(initial(2, hidden), xs, Y), xs, Y)} errors")

print("== 2. The impossible flowers ==")
print(f"same petal, different species: {twins()}")

print("== 3. Measuring on flowers the network has not seen ==")
test = [i for i in range(len(Y)) if i % 5 == 0]  # one flower in five, 20 in total
fit = [i for i in range(len(Y)) if i % 5]
for name, columns in (("petal", PETAL), ("all four", ALL)):
    xs = standardised(columns)
    net = train(initial(len(columns), 2), [xs[i] for i in fit], [Y[i] for i in fit])
    print(
        f"{name}: {errors(net, [xs[i] for i in fit], [Y[i] for i in fit])} errors in the 80 it learned from, "
        f"{errors(net, [xs[i] for i in test], [Y[i] for i in test])} in the 20 it never saw"
    )
