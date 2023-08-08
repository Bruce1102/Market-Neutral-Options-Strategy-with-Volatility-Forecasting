import numpy as np
from scipy.optimize import minimize


class JDMSimulator:
    def __init__(self, initial_price: float=1, drift: float=0.001, volatility: float=0.5, lambda_jump: float=0, jump_mean: float=0.1, 
                 jump_std: float=0.5, p: float=0.5, eta1: float=0.03, eta2: float=0.03, model: str="kou", 
                 dt: float=1/252) -> None:
        """
        Initialize the Jump Diffusion Model Simulator.

        Parameters:
        - initial_price: Initial asset price.
        - drift: Drift term.
        - volatility: Volatility of the asset.
        - lambda_jump: Intensity of the jump.
        - jump_mean: Mean of the jump size.
        - jump_std: Standard deviation of the jump size.
        - p: Probability for positive jump in Kou model.
        - eta1, eta2: Parameters for the exponential distribution in Kou model.
        - model: Type of model ("merton" or "kou").
        - dt: Time step.
        """
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        self.model = model
        self.dt = dt

    def get_params(self):
        """Prints and returns a tuple of parameters"""
        print(f"Initial Price: {self.initial_price}",
              f"Drift: {self.drift}",
              f"Volatility: {self.volatiltiy}",
              f"Lambda Jump: {self.lambda_jump}",
              f"Jump Mean: {self.jump_mean}",
              f"Jump Std: {self.jump_std}",
              f"Model: {self.model}")
        
        return (self.initial_price, self.drift, self.volatility, 
                self.lambda_jump, self.jump_mean, self.jump_std, 
                self.model)

    def simulate(self, n_days: int, initial_price: float = None):
        """Simulate asset prices using the Jump Diffusion Model."""
        prices = []
        if initial_price != None:
            prices.append(initial_price)
        else:
            prices.append(self.initial_price)

        for _ in range(n_days):
            drift_term = self.drift * prices[-1] * self.dt
            shock_term = self.volatility * prices[-1] * np.sqrt(self.dt) * np.random.normal()
            jump = self._jump_term()
            
            next_price = prices[-1] + drift_term + shock_term + jump
            prices.append(next_price)
        return np.array(prices)

    def _jump_term(self):
        """Compute the jump term based on the model."""
        jump = 0
        if np.random.rand() < self.lambda_jump * self.dt:
            if self.model == "merton":
                jump = np.random.normal(self.jump_mean, self.jump_std)
            elif self.model == "kou":
                jump = self.eta1 * np.random.exponential() if np.random.rand() < self.p else -self.eta2 * np.random.exponential()
        return jump

    @staticmethod
    def negative_log_likelihood(params, prices, model):
        """Negative log likelihood function for parameter estimation."""
        drift, volatility, lambda_jump, jump_mean, jump_std, p, eta1, eta2 = params
        simulator = JDMSimulator(prices[0], drift, volatility, lambda_jump, jump_mean, jump_std, p, eta1, eta2, model)
        simulated_prices = simulator.simulate(len(prices) - 1, prices[0])
        log_returns = np.log(simulated_prices[1:] / simulated_prices[:-1])
        log_returns_real = np.log(prices[1:] / prices[:-1])
        return np.sum((log_returns - log_returns_real)**2)

    def estimate_parameters(self, prices):
        """Estimate model parameters and update the instance's parameters."""
        init_params = [self.drift, self.volatility, self.lambda_jump, self.jump_mean, self.jump_std, self.p, self.eta1, self.eta2]

        bounds = [(None, None), (0.0001, 2), (0, None), (None, None), (0, 1), (0, 1), (0, 1), (0, 1)]
        result = minimize(self.negative_log_likelihood, init_params, args=(prices, self.model), bounds=bounds)
        
        # Update the instance's parameters
        self.initial_price = prices[0]
        self.drift, self.volatility, self.lambda_jump, self.jump_mean, self.jump_std, self.p, self.eta1, self.eta2 = result.x
        
        return result.x


class CIRJumpModel:
    """
    The CIRJumpModel class simulates interest rates using a combination of the Cox-Ingersoll-Ross (CIR) model 
    and Merton's jump diffusion model. The model captures both the mean-reverting behavior of interest rates 
    and potential sudden jumps due to unexpected market events.
    
    Attributes:
    - r (array-like): Historical interest rate data.
    - a (float): Speed of mean reversion.
    - b (float): Long-term mean level of the interest rate.
    - sigma (float): Volatility of the interest rate.
    - lambda_jump (float): Intensity of the jump process.
    - jump_mean (float): Mean of the jump size.
    - jump_std (float): Standard deviation of the jump size.
    - dt (float): Time step for simulation.
    """
    
    def __init__(self, initial_rate: float=0.01, a: float=0.1, b: float=0.05, sigma: float=0.1, lambda_jump: float=0.1, 
                 jump_mean: float=0, jump_std: float=0.01, dt: float=1/252) -> None:
        """
        Initialize the CIRJumpModel with given parameters.
        
        Parameters:
        - initial_rate (float, optional): initial interest rate.
        - a (float, optional): Speed of mean reversion. Defaults to 0.1.
        - b (float, optional): Long-term mean level of the interest rate. Defaults to 0.05.
        - sigma (float, optional): Volatility of the interest rate. Defaults to 0.01.
        - lambda_jump (float, optional): Intensity of the jump process. Defaults to 0.1.
        - jump_mean (float, optional): Mean of the jump size. Defaults to 0.
        - jump_std (float, optional): Standard deviation of the jump size. Defaults to 0.01.
        - dt (float, optional): Time step for simulation. Defaults to 1/252 (assuming 252 trading days in a year).
        """
        self.initial_rate = initial_rate
        self.a = a
        self.b = b
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.dt = dt

    def simulate(self, n_days: int, initial_rate: float = None) -> np.array:
        """
        Simulate interest rates for a given number of days using the CIR Jump model.
        
        Parameters:
        - n_days (int): Number of days to simulate.
        
        Returns:
        - numpy.array: Simulated interest rates.
        """
        rates = []
        if initial_rate != None:
            rates.append(initial_rate)
        else:
            rates.append(self.initial_rate)

        for _ in range(n_days):
            dr = self.a * (self.b - rates[-1]) * self.dt + self.sigma * np.sqrt(rates[-1] * self.dt) * np.random.normal()
            jump = self._jump_term()
            rates.append(rates[-1] + dr + jump)
        return np.array(rates)

    def _jump_term(self) -> None:
        """
        Compute the jump term based on the jump parameters.
        
        Returns:
        - float: Jump term for the current time step.
        """
        jump = 0
        if np.random.rand() < self.lambda_jump * self.dt:
            jump = np.random.normal(self.jump_mean, self.jump_std)
        return jump

    @staticmethod
    def negative_log_likelihood(params: list, rates: float) -> float:
        """
        Compute the negative log likelihood for the CIR Jump model given parameters and observed rates.
        
        Parameters:
        - params (tuple): Model parameters (a, b, sigma).
        - rates (np.array): Observed interest rates.
        
        Returns:
        - float: Negative log likelihood value.
        """
        a, b, sigma = params
        dt = 1/252
        n = len(rates)
        likelihoods = []
        for t in range(1, n):
            mu = a * (b - rates[t-1]) * dt
            sigma_t = sigma * np.sqrt(rates[t-1] * dt)
            likelihood = (1 / (sigma_t * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((rates[t] - rates[t-1] - mu) / sigma_t)**2)
            likelihoods.append(np.log(likelihood))
        return -sum(likelihoods)

    def estimate_parameters(self, rates: list)-> list:
        """
        Estimate model parameters from observed interest rates using maximum likelihood estimation.
        
        Returns:
        - np.array: Estimated model parameters (a, b, sigma).
        """
        initial_params = [self.a, self.b, self.sigma]
        bounds = [(0, None), (0, None), (0.000000000000000001, None)]
        result = minimize(self.negative_log_likelihood, initial_params, args=(rates), bounds=bounds)
        self.a, self.b, self.sigma = result.x
        self.initial_rate = rates[0]
        return result.x


class CorrelatedJDMSimulator:
    """
    A class to simulate stock prices using a correlated jump diffusion model.
    """
    def __init__(self, initial_prices: float=1, drifts: float=0, volatilities: float=0, lambda_jumps: float=0, 
                 jump_means: float=0, jump_stds: float=0, correlation_matrix: np.ndarray=np.ones((1,1)), dt: float=1/252) -> None:
        """
        Initialize the simulator with model parameters.
        
        Parameters:
        - initial_prices: Initial stock prices.
        - drifts: Drift terms for each stock.
        - volatilities: Volatility terms for each stock.
        - lambda_jumps: Jump intensities for each stock.
        - jump_means: Mean of jumps for each stock.
        - jump_stds: Standard deviation of jumps for each stock.
        - correlation_matrix: Correlation matrix for stock returns.
        - dt: Time step for simulation (default is 1/252, assuming 252 trading days in a year).
        """
        self.initial_prices = initial_prices
        self.drifts = drifts
        self.volatilities = volatilities
        self.lambda_jumps = lambda_jumps
        self.jump_means = jump_means
        self.jump_stds = jump_stds
        self.correlation_matrix = correlation_matrix
        self.dt = dt
        self.L = np.linalg.cholesky(correlation_matrix)

    def simulate(self, n_days: int, initial_prices: np.array = []) -> list:
        """
        Simulate stock prices for a given number of days.
        
        Parameters:
        - n_days: Number of days to simulate.
        
        Returns:
        - prices: Simulated stock prices.
        """
        num_stocks = len(self.initial_prices)
        prices = np.zeros((n_days, num_stocks))

        if len(initial_prices) == 0:
            prices[0] = self.initial_prices
        else:
            prices[0] = initial_prices

        for t in range(1, n_days):
            correlated_normals = np.dot(self.L, np.random.normal(size=num_stocks))
            for i in range(num_stocks):
                drift_term = self.drifts[i] * prices[t-1, i] * self.dt
                shock_term = self.volatilities[i] * prices[t-1, i] * np.sqrt(self.dt) * correlated_normals[i]
                jump = 0
                if np.random.rand() < self.lambda_jumps[i] * self.dt:
                    jump = np.random.normal(self.jump_means[i], self.jump_stds[i])
                prices[t, i] = prices[t-1, i] + drift_term + shock_term + jump

        return prices

    @staticmethod
    def negative_log_likelihood(params: list, prices: float, num_stocks: int) -> float:
        """
        Compute the negative log likelihood for the model given parameters and observed prices.
        
        Parameters:
        - params: Model parameters.
        - prices: Observed stock prices.
        - num_stocks: Number of stocks.
        
        Returns:
        - Negative log likelihood value.
        """
        drifts = params[:num_stocks]
        volatilities = params[num_stocks:2*num_stocks]
        lambda_jumps = params[2*num_stocks:3*num_stocks]
        jump_means = params[3*num_stocks:4*num_stocks]
        jump_stds = params[4*num_stocks:5*num_stocks]
        simulator = CorrelatedJDMSimulator(prices[0], drifts, volatilities, lambda_jumps, jump_means, jump_stds, np.identity(num_stocks))
        simulated_prices = simulator.simulate(len(prices) - 1)
        simulated_prices = np.vstack([prices[0], simulated_prices])  # Add the initial prices to the simulated prices
        log_returns = np.log(simulated_prices[1:] / simulated_prices[:-1])
        log_returns_real = np.log(prices[1:] / prices[:-1])
        return np.sum((log_returns - log_returns_real)**2)


    def estimate_parameters(self, prices: list) -> list:
        """
        Estimate model parameters from observed stock prices.
        
        Parameters:
        - prices: Observed stock prices.
        
        Returns:
        - Estimated model parameters.
        """
        num_stocks = prices.shape[1]
        init_params = [0.0001]*num_stocks + [0.2]*num_stocks + [0.5]*num_stocks + [0]*num_stocks + [0.2]*num_stocks
        bounds = [(None, None)]*num_stocks + [(0.01, 3)]*num_stocks + [(0, 1)]*num_stocks + [(None, None)]*num_stocks + [(0, 3)]*num_stocks
        result = minimize(self.negative_log_likelihood, init_params, args=(prices, num_stocks), bounds=bounds)
        
        # Update parameters:
        estimated_params = result.x
        self.initial_prices = prices[0]
        self.drifts = estimated_params[:num_stocks]
        self.volatilities = estimated_params[num_stocks:2*num_stocks]
        self.lambda_jumps = estimated_params[2*num_stocks:3*num_stocks]
        self.jump_means = estimated_params[3*num_stocks:4*num_stocks]
        self.jump_stds = estimated_params[4*num_stocks:5*num_stocks]

         # Compute the correlation matrix of the log returns of the observed prices
        log_returns = np.log(prices[1:] / prices[:-1])
        data = log_returns.T
        cov_matrix = np.cov(data)
        std_devs = np.sqrt(np.diag(cov_matrix))
        self.correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
        self.L = np.linalg.cholesky(self.correlation_matrix)

        return estimated_params
    

