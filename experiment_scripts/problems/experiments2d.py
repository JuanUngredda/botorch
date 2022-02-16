import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
except:
    pass


class function2d:
    """
    This is a benchmark of bi-dimensional functions interesting to optimize.

    """

    def plot(self):
        x1 = torch.linspace(self.lb[0], self.ub[0], 100)
        x2 = torch.linspace(self.lb[1], self.ub[1], 100)
        X1, X2 = torch.meshgrid(x1, x2)
        X = torch.hstack((X1.reshape(100 * 100, 1), X2.reshape(100 * 100, 1)))
        Y = torch.Tensor([self.f(x) for x in X])

        fig = plt.figure()

        ax = fig.gca(projection="3d")
        ax.plot_surface(
            X1,
            X2,
            Y.reshape((100, 100)),
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title(self.problem)

        plt.figure()
        plt.contourf(X1, X2, Y.reshape((100, 100)), 100)
        plt.colorbar()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(self.problem)
        plt.show()

    def __call__(self, X):
        return self.f(X)

class rosenbrock(function2d):
    """
    Cosines function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-0.5, -1.5])
        self.ub = torch.Tensor([3, 2])
        self.bounds = torch.vstack([self.lb, self.ub])
        self.min = torch.Tensor([0, 0])
        self.fmin = 0
        self.problem = "Rosenbrock"

    def f(self, X):
        X = X.squeeze()
        fval = 100 * (X[1] - X[0] ** 2) ** 2 + (X[0] - 1) ** 2
        return -fval


class beale(function2d):
    """
    Cosines function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-1, -1])
        self.ub = torch.Tensor([1, 1])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.Tensor([0, 0])
        self.fmin = 0
        self.problem = "beale"

    def f(self, X):
        X = X.squeeze()
        fval = 100 * (X[1] - X[0] ** 2) ** 2 + (X[0] - 1) ** 2
        return -fval


class dropwave(function2d):
    """
    Cosines function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-1, -1])
        self.ub = torch.Tensor([1, 1])
        self.bounds = torch.vstack([self.lb, self.ub])
        self.min = torch.Tensor([0, 0])
        self.fmin = 0
        self.problem = "dropwave"

    def f(self, X):
        X = X.squeeze()
        fval = -(1 + torch.cos(12 * torch.sqrt(X[0] ** 2 + X[1] ** 2))) / (
                0.5 * (X[0] ** 2 + X[1] ** 2) + 2
        )
        return -fval


class cosines(function2d):
    """
    Cosines function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([0, 0])
        self.ub = torch.Tensor([1, 1])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.Tensor([0.31426205, 0.30249864])
        self.fmin = -1.59622468
        self.problem = "Cosines"

    def f(self, X):
        X = X.squeeze()
        u = 1.6 * X[0] - 0.5
        v = 1.6 * X[1] - 0.5
        fval = 1 - (
                u ** 2
                + v ** 2
                - 0.3 * torch.cos(3 * torch.pi * u)
                - 0.3 * torch.cos(3 * torch.pi * v)
        )
        return -fval


class branin(function2d):
    """
    Branin function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, a=None, b=None, c=None, r=None, s=None, t=None, **kwargs):
        self.input_dim = 2

        self.lb = torch.Tensor([-5, 1])
        self.ub = torch.Tensor([10, 15])
        self.bounds = torch.vstack([self.lb, self.ub])

        if a == None:
            self.a = 1
        else:
            self.a = a
        if b == None:
            self.b = 5.1 / (4 * torch.pi ** 2)
        else:
            self.b = b
        if c == None:
            self.c = 5 / torch.pi
        else:
            self.c = c
        if r == None:
            self.r = 6
        else:
            self.r = r
        if s == None:
            self.s = 10
        else:
            self.s = s
        if t == None:
            self.t = 1 / (8 * torch.pi)
        else:
            self.t = t

        self.min = torch.Tensor(
            [[-torch.pi, 12.275], [torch.pi, 2.275], [9.42478, 2.475]]
        )
        self.fmin = 0.397887
        self.problem = "Branin"

    def f(self, X):
        X = X.squeeze()
        x1 = X[0]
        x2 = X[1]
        fval = (
                self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.r) ** 2
                + self.s * (1 - self.t) * torch.cos(x1)
                + self.s
        )

        return -fval


class goldstein(function2d):
    """
    Goldstein function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2

        self.lb = torch.Tensor([-2, -2])
        self.ub = torch.Tensor([2, 2])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.tensor([0, -1])
        self.fmin = 3

        self.problem = "Goldstein"

    def f(self, X):
        X = X.squeeze()
        x1 = X[0]
        x2 = X[1]
        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b
        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b
        fval = fact1 * fact2
        return -fval


class sixhumpcamel(function2d):
    """
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-2, -1])
        self.ub = torch.Tensor([2, 1])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.Tensor([[0.0898, -0.7126], [-0.0898, 0.7126]])
        self.fmin = -1.0316
        self.problem = "Six-hump camel"

    def f(self, x):
        x = x.squeeze()
        x1 = x[0]
        x2 = x[1]
        term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
        fval = term1 + term2 + term3
        return -fval


class mccormick(function2d):
    """
    Mccormick function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-1.5, -3])
        self.ub = torch.Tensor([4, 4])
        self.bounds = torch.vstack([self.lb, self.ub])
        self.min = torch.Tensor([[-0.54719, -1.54719]])
        self.fmin = -1.9133
        self.problem = "Mccormick"

    def f(self, x):
        x = x.squeeze()
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = torch.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        fval = term1 + term2 + term3 + term4 + 1
        return -fval


class powers(function2d):
    """
    Powers function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-1, -1])
        self.ub = torch.Tensor([1, 1])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.tensor([[0, 0]])
        self.fmin = 0
        self.problem = "Sum of Powers"

    def f(self, x):
        x = x.squeeze()
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = abs(x1) ** 2 + abs(x2) ** 3
        return -fval


class eggholder:
    def __init__(self, **kwargs):
        self.input_dim = 2
        self.lb = torch.Tensor([-512, -512])
        self.ub = torch.Tensor([512, 512])
        self.bounds = torch.vstack([self.lb, self.ub])

        self.min = torch.Tensor([[512, 404.2319]])
        self.fmin = -959.6407

        self.problem = "Egg-holder"

    def f(self, X):
        X = X.squeeze()
        x1 = X[0]
        x2 = X[1]
        fval = -(x2 + 47) * torch.sin(
            torch.sqrt(abs(x2 + x1 / 2 + 47))
        ) + -x1 * torch.sin(torch.sqrt(abs(x1 - (x2 + 47))))

        return -fval
