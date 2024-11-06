import random
import numpy as np


class ParameterRange:
    def __init__(self, choices, discrete=None):
        """
        choices: eiter a list of possible values or a tuple of (lower, upper) bounds
        k: number of values to sample
        discrete: choices are choices or bounds
        dtype: int, float or str
        """
        self.dtype = None
        if all(map(lambda x: isinstance(x, int), choices)):
            self.dtype = int
        elif any(map(lambda x: isinstance(x, float), choices)):
            self.dtype = float
        elif all(map(lambda x: isinstance(x, str), choices)):
            self.dtype = str
        if discrete is None:
            if len(choices) != 2 or self.dtype == str:
                discrete = True
            else:
                discrete = False
        self.choices = choices
        self.discrete = discrete
        if not discrete:
            if len(choices) != 2:
                raise ValueError(
                    "Continuous parameters must have exactly two choices: lower and upper bound"
                )
            if self.dtype not in [int, float]:
                raise ValueError("Continuous parameters must have dtype int or float")

    def info(self):
        return dict(
            choices=self.choices,
            discrete=self.discrete,
            dtype=str(self.dtype),
        )

    def log_normalize(self, value):
        low, high = self.choices
        return (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))

    def random(self, k=None, log_sample=False):
        if self.discrete:
            if k == None:
                return random.choice(self.choices)
            else:
                return random.choices(self.choices, k=k)
        else:
            if k == None:
                if self.dtype == float:
                    if log_sample:
                        return np.exp(
                            np.log(self.choices[0])
                            + random.random()
                            * (np.log(self.choices[1]) - np.log(self.choices[0]))
                        )
                    return random.uniform(*self.choices)
                elif self.dtype == int:
                    if log_sample:
                        raise ValueError("Cannot log sample integer values")
                    return random.randint(*self.choices)
            else:
                result = []
                for _ in range(k):
                    if self.dtype == float:
                        if log_sample:
                            result.append(
                                np.exp(
                                    np.log(self.choices[0])
                                    + random.random()
                                    * (
                                        np.log(self.choices[1])
                                        - np.log(self.choices[0])
                                    )
                                )
                            )
                        else:
                            result.append(random.uniform(*self.choices))
                    elif self.dtype == int:
                        if log_sample:
                            raise ValueError("Cannot log sample integer values")
                        result.append(random.randint(*self.choices))
                return result
