import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, poisson


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
            jump = self._jump_term()*prices[-1]
            
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


    def _drift_diffusion_pdf(self, R: np.array, mu: float, sigma: float, dt: float) -> float:
        """Compute the pdf of the drift-diffusion component of the model"""
        return norm.pdf(R, (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt))
    

    def _jump_pdf(self, R: np.array, lambda_jump: float, jump_mean: float, jump_std: float,
                  p: float, eta1: float, eta2: float, dt: float, max_jumps: int=11) -> np.array:
        """Compute the pdf of the jump component of the model"""
        
        if self.model == 'merton':
            # Computing pdf of jump
            poisson_probs = np.array([poisson.pmf(k, lambda_jump * dt) for k in range(max_jumps)])
            # Computing pdf of size of jump
            normal_pdfs = np.array([norm.pdf(R, k * jump_mean * dt, np.sqrt(k * jump_std**2 * dt)) for k in range(max_jumps)])
            
            # Compute the weighted sum of the Normal PDFs
            return np.sum(poisson_probs[:, np.newaxis] * normal_pdfs, axis=0)

        elif self.model == 'kou':
            # Double exponential distribution for the jump sizes
            jump_size = p * eta1 * np.exp(-eta1 * R) * (R > 0) + (1-p) * eta2 * np.exp(eta2 * R) * (R < 0)
            # Compute the weighed sum of the poisson pdf
            combined_pdf = np.sum([poisson.pmf(k, lambda_jump * dt)*jump_size for k in range(max_jumps)], axis=0)

            return combined_pdf

    def _neg_log_likelihood(self, params, R, dt):
        """Negative log-likelihood function for the Jump model."""
        mu, sigma, lambda_jump, jump_mean, jump_std, p, eta1, eta2,  = params
   
        # Computing likelihood for drift-diffusion and jump component
        dd_pdf   = self._drift_diffusion_pdf(R, mu, sigma, dt)
        jump_pdf = self._jump_pdf(R, lambda_jump, jump_mean, jump_std, p, eta1, eta2, dt)
        
        # Combining them together
        combined_pdf = dd_pdf * jump_pdf
        return -np.sum(np.log(combined_pdf))

    def estimate_parameters(self, x):
        """Estimate parameters using MLE and update the model parameters."""
        # computing log returns
        R = np.diff(np.log(x))

        # Initial testing parameters
        params0 = [self.drift, self.volatility, self.lambda_jump, 
                   self.jump_mean, self.jump_std, self.p, self.eta1, self.eta2]
        # Setting boundaries
        bounds = [(None, None), (0.001, None), (0, 1), (None, None), (0.001, None), (0, 1), (0, None), (0, None)]

        # Numerically solving MLE
        result = minimize(self._neg_log_likelihood, params0, args=(R, self.dt), bounds=bounds)
        
        self.initial_price = x[0]
        self.drift, self.volatility, self.lambda_jump, self.jump_mean, self.jump_std, self.p, self.eta1, self.eta2 = result.x
        return result.x


class CIRJumpModel:
    """
    The CIRJumpModel class simulates interest rates using a combination of the Cox-Ingersoll-Ross (CIR) model 
    and Merton's jump diffusion model. The model captures both the mean-reverting behavior of interest rates 
    and potential sudden jumps due to unexpected market events.
    
    Attributes:
    - r (array-like): Historical interest rate data.
    - k (float): Speed of mean reversion.
    - theta (float): Long-term mean level of the interest rate.
    - sigma (float): Volatility of the interest rate.
    - lambda_jump (float): Intensity of the jump process.
    - jump_mean (float): Mean of the jump size.
    - jump_std (float): Standard deviation of the jump size.
    - p: Probability for positive jump in Kou model.
    - eta1, eta2: Parameters for the exponential distribution in Kou model.
    - model: Type of model ("merton" or "kou").
    - dt (float, optional): Time step for simulation. Defaults to 1/252 (assuming 252 trading days in a year).
    """
    
    def __init__(self, initial_rate: float=0.01, k: float=0.1, theta: float=0.05, sigma: float=0.1, lambda_jump: float=0, jump_mean: float=0.1, 
                 jump_std: float=0.5, p: float=0.5, eta1: float=0.03, eta2: float=0.03, model: str="kou", dt: float=1/252) -> None:
        """ Initialize the CIRJumpModel with given parameters."""

        self.initial_rate = initial_rate
        self.k = k
        self.thetat = theta
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        self.model = model
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
        k, theta, sigma = params
        dt = 1/252
        n = len(rates)
        likelihoods = []
        for t in range(1, n):
            mu = k * (theta - rates[t-1]) * dt
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
        initial_params = [self.k, self.theta, self.sigma]
        bounds = [(0, None), (0, None), (0.000000000000000001, None)]
        result = minimize(self.negative_log_likelihood, initial_params, args=(rates), bounds=bounds)
        self.k, self.theta, self.sigma = result.x
        self.initial_rate = rates[0]
        return result.x


class CorrelatedJDMSimulator:
    """
    A class to simulate stock prices using a correlated jump diffusion model.
    """
    def __init__(self, initial_prices: float=1, drifts: float=1, volatilities: float=0, lambda_jumps: float=0, 
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
    

"""
EXAMPLE USE:

from stoch_modelling import CIRJumpModel

#S = time series dataset of asset S
#n_days = number of days to simulate


# Initialise CIR Jump model:
cirjump_model = CIRJumpModel()


# Fitting model with MLE:
cirjump_model.fit(S)

# Run simulations:
# Setting initial_rate is optional, by default it will be set to S[0]
sim_results = simulate(n_days, initial_rate = S[-1])



"""