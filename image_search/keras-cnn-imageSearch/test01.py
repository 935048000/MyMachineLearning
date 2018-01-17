from memory_profiler import profile


@profile (precision=6)
def primes(n):
    I = 0
    J = []
    for i in range(n):
        I += i
        J.append(i)
    return J


primes (100000)

