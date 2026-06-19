import math

def cochran_sample(df, confidence_z=1.96, p=0.5, error=0.05):
    N = len(df)

    n0 = (confidence_z**2 * p * (1 - p)) / (error**2)

    n = n0 / (1 + ((n0 - 1) / N))

    sample_size = math.ceil(n)

    print(f"Population size: {N}")
    print(f"Sample size: {sample_size}")

    return df.sample(n=sample_size, random_state=42)