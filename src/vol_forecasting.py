import numpy as np
import pandas as pd
from arch import arch_model
from src.stoch_modelling import CorrelatedJDMSimulator, JDMSimulator, CIRJumpModel
from src.linear_factor_model import LinearFactorModel


# Constants
CORR_JDM_FACTORS = ["AAPL_Close", "MSFT_Close", "AMZN_Close"]
LFM_FACTORS = ["AAPL_Close", "MSFT_Close", "AMZN_Close", "VIX_Close", "IRX_Close"]
    
    
class GARCHVolForecast:
    """
    Compute the GARCH(p, q) volatility forecast for n days into the future.

    Attributes:
    - series (np.array): The time series data to be modeled.
    - p (int, optional): The lag order of the GARCH model. Default is 1.
    - q (int, optional): The order of the moving average. Default is 1.
    - n (int, optional): Number of days forecasted into the future. Default is 7.

    Methods:
    fit_models():
        Initializes and estimates parameters for the models using provided data.
    
    forecast_volatility(n_days: int, n_simulations: int = None) -> float:
        Forecast the portfolio's volatility using Monte Carlo simulations for a given number of days.
    """
    def __init__(self, data: pd.DataFrame, p: int=1, q: int=1, n: int=7, target: str='SPX_Close') -> None:
        """Initializes the GARCHVolForecast with data and factors."""

        self.data   = data
        self.p      = p
        self.q      = q
        self.n      = n
        self.target = target

        self.model = None

    def fit_model(self) -> None:
        """
        Initialises the GARCH model, and fits it with the provided data

        Note:
        This method will update the object's model attributes.
        """
        model = arch_model(self.data[self.target], vol='Garch', p=self.p, q=self.q, rescale=False)
        model = model.fit(disp='off')

        self.model = model

    def forecast_volatility(self, n_days: int) -> float:
        """
        Forecast volatility using the trained GARCH model.
        Parameters:
            - n_days: Number of days to simulate.

        Returns:
            - Forecasted volatility value on n_days'th day.
        """
        forecast = self.model.forecast(start=len(self.data[self.target])-1, reindex = False, horizon=n_days)

        volatilities = np.sqrt(forecast.variance.iloc[-1, :n_days].values)

        return volatilities[-1]
    

class StochasticVolForecast:
    """
    A class to forecast volatility using various stochastic models.

    Attributes:
    - data (pd.DataFrame)           :  The dataset containing the required columns for analysis.
    - lfm_factors (list)            :  List of factors/columns required for Linear Factor Model.
    - corr_jdm_factors (list)       :  List of factors/columns for Correlated Jump Diffusion Model.
    - jdm_factor (str)              : default='VIX_Close' Factor/column for Jump Diffusion Model.
    - cirj_factor (str)             : default='IRX_Close' Factor/column for CIR Jump Model.
    - lfm_target (str)              : default='SPX_Close' Target column for Linear Factor Model.
    - initial_price (float)         : Initial price extracted from the dataset for the lfm_target.
    - corr_jdm_model (Model object) : Model object for Correlated Jump Diffusion.
    - jdm_model (Model object)      : for Jump Diffusion.
    - cirj_model (Model object)     : for CIR Jump.
    - lfm_model (Model object)      : Model object for Linear Factor Model.

    Methods:
    fit_models():
        Initializes and estimates parameters for the models using provided data.
    
    simulate_portfolio_path(n_days: int) -> np.ndarray:
        Simulate portfolio path using the estimated models for a given number of days.

    forecast_volatility(n_days: int, n_simulations: int = None) -> float:
        Forecast the portfolio's volatility using Monte Carlo simulations for a given number of days.
    """
    def __init__(self, data: pd.DataFrame, lfm_factors: str=LFM_FACTORS, corr_jdm_factors: str=LFM_FACTORS, jdm_factor: str='VIX_Close', 
                 cirj_factor: str='IRX_Close', lfm_target: str='SPX_Close'):

        """Initializes the StochasticVolForecast with data and factors."""
        # Data attributes
        self.data             = data
        self.lfm_factors      = lfm_factors
        self.corr_jdm_factors = corr_jdm_factors
        self.jdm_factor       = jdm_factor
        self.cirj_factor      = cirj_factor
        self.lfm_target       = lfm_target

        # Model attributes
        self.corr_jdm_model = None
        self.jdm_model      = None
        self.cirj_model     = None
        self.lfm_model      = None

        # Constants
        self.initial_price = data[lfm_target].iloc[-1]

    def fit_model(self):
        """
        Initializes the various models and estimates their parameters using the provided data.
        
        Note:
        This method will update the object's model attributes.
        """
        # Initialise models
        corr_jdm_model = CorrelatedJDMSimulator()
        jdm_model      = JDMSimulator(model = "Merton")
        cirj_model     = CIRJumpModel()
        lfm_model      = LinearFactorModel()

        # Estimating parameters for models
        corr_jdm_model.estimate_parameters(self.data[CORR_JDM_FACTORS].values)
        jdm_model.estimate_parameters(self.data[self.jdm_factor].values)
        cirj_model.estimate_parameters(self.data[self.cirj_factor].values)
        lfm_model.fit(self.data[self.lfm_target].values, self.data[self.lfm_factors].values)

        # set initial price
        corr_jdm_model.initial_prices = self.data[CORR_JDM_FACTORS].iloc[0]
        jdm_model.initial_price = self.data[self.jdm_factor].iloc[0]
        cirj_model.initial_rate = self.data[self.cirj_factor].iloc[0]

        # setting object attributes
        self.corr_jdm_model = corr_jdm_model
        self.jdm_model      = jdm_model
        self.cirj_model     = cirj_model
        self.lfm_model      = lfm_model

    def simulate_portfolio_path(self, n_days: int) -> np.ndarray:
        """
        Simulates the portfolio path for a given number of days.

        Parameters:
        - n_days (int): Number of days to simulate.

        Returns:
        - np.ndarray: Array containing the simulated portfolio path values.
        """
        # Simulating paths for independent models
        corr_jdm_sim = self.corr_jdm_model.simulate(n_days)
        jdm_sim      = self.jdm_model.simulate(n_days-1).reshape(-1,1)
        cirj_sim     = self.cirj_model.simulate(n_days-1).reshape(-1,1)
        
        # concatenating simulations
        sim_concat = np.hstack((corr_jdm_sim, jdm_sim, cirj_sim))

        # applying LFM model to produce final output
        return self.lfm_model.predict(sim_concat)


    def forecast_volatility(self, n_days: int, n_simulations: int=10_000) -> float:
        """
        Forecast volatility using Monte Carlo simulations.
        Parameters:
            - n_days: Number of days to simulate.
            - n_simulations: Number of simulations to run. Defaults to 10,000.

        Returns:
            - Forecasted volatility value on n_days'th day.
        """

        # Initialising list to save the forecasted volatilities
        volatilities = []
        
        # running Monte-Carlo simulation to compute volatility
        for _ in range(n_simulations):
            # Simulating path
            portfolio_path = self.simulate_portfolio_path(n_days)

            # Computing returns
            returns = np.diff(np.log(portfolio_path))

            # Computing volatility and appending to list
            volatility = np.std(returns)
            volatilities.append(volatility)

        return np.mean(volatilities) * self.initial_price
