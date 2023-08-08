import numpy as np
import math
from scipy.stats import norm

class Option:
    """
    Represents a European option using the Black-Scholes model.
    
    Attributes:
        S (float): Current stock (or spot) price.
        K (float): Option strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free rate.
        sigma (float): Volatility of the underlying stock.
        type (str): Option type ("call" or "put").
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, type: str, long_short: str):
        """
        Initializes a new instance of the Option class.

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to expiration (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            type (str): Type of the option - either "call" or "put".
        """

        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
    
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.type = type  # "call" or "put"
        self.long_short = long_short # "long" or "short"
        self.initial_value = self.current_value(T, S)
    
    def get_params(self) -> dict:
        params = {'S': self.S,
                  'K': self.K,
                  'T': self.T,
                  'r': self.r,
                  'sigma': self.sigma,
                  'initial_value': self.initial_value,
                  'type': self.type,
                  'long/short': self.long_short}
        return params

    def current_value(self, time_to_maturity: float, spot_price: float) -> float:
        """
        Computes and returns the current value of the option using the Black-Scholes formula given a specific time to maturity and spot price.

        Args:
            time_to_maturity (float): Time to option's expiration (in years).
            spot_price (float): Current price of the underlying asset.

        Returns:
            float: Current value of the option.
        """

        d1 = (math.log(spot_price / self.K) + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity) / (self.sigma * math.sqrt(time_to_maturity))
        d2 = d1 - self.sigma * math.sqrt(time_to_maturity)
        cur_val = None
        if self.type == "call":
            cur_val = spot_price * norm.cdf(d1) - self.K * math.exp(-self.r * time_to_maturity) * norm.cdf(d2)
        elif self.type == "put":
            cur_val = self.K * math.exp(-self.r * time_to_maturity) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        if self.long_short == 'long':
            return cur_val
        if self.long_short == 'short':
            return -cur_val

    def pnl_at_expiry(self, spot_at_expiry: float) -> float:
        """
        Compute the PnL at expiry for the option.

        Args:
        - spot_at_expiry (float): Spot price at the time of option expiry.

        Returns:
        - float: PnL at expiry.
        """
        pnl = None
        if self.type == "call":
            intrinsic_value = max(0, spot_at_expiry - self.K)
        else:  # put option
            intrinsic_value = max(0, self.K - spot_at_expiry)

        # Computing pnl considering long_short
        if self.long_short == 'long':
            return intrinsic_value - self.initial_value
        
        elif self.long_short == 'short':
            return -self.initial_value - intrinsic_value

    def get_initial_value(self) -> float:
        """
        Returns the initial value of the option.

        Returns:
        - float: The initial value of the option
        """
        return self.initial_value


class OptionCombination:
    """
    Represents a generic combination of options. 
    
    This class can be utilized as a base class for more specific option combinations 
    like Straddles, Strangles, etc.

    Attributes:
        options (list of Option): A list of Option objects that form the option combination.
    """
    def __init__(self, options):
        """
        Initializes a new instance of the OptionCombination class.

        Args:
            options (list of Option): A list of Option objects.
        """
        self.combination = ''
        self.options = options
        self.option_values  = [option.get_initial_value() for option in self.options]
        self.option_types   = [option.get_params()["type"] for option in self.options]
        self.option_strikes = [option.get_params()["K"] for option in self.options]
        self.option_initial_value = sum(self.option_values)




    def get_options(self) -> list:
        """
        Returns a list of Option objects that form the option combination.

        Returns:
            list: List of Option objects.
        """
        return self.options
    
    def get_option_values(self) -> list:
        """
        Returns a list of Option values from the option combination.

        Returns:
            list: List of Option values.
        """
        return self.option_values

    def get_option_types(self) -> list:
        """
        Returns a list of Option types (call/put) from the option combination.

        Returns:
            list: List of Option values.
        """
        return self.option_types

    def get_option_strikes(self) -> list:
        """
        Returns a list of Option strikes from the option combination.

        Returns:
            list: List of Option values.
        """
        return self.option_strikes

    def get_initial_value(self) -> float:
        """
        Returns the initial value of the option combination.

        Returns:
        - float: The initial value of the option
        """
        return self.option_initial_value
    
    def _opposite_long_short(long_short):
        """
        Returns the opposite of the given "long_short" value.

        Args:
            long_short (str): A string that can be either "long" or "short".

        Returns:
            str: The opposite of the given "long_short" value.
        """
        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
        return 'long' if long_short == 'short' else 'short'

    def current_value(self, time_to_maturity: float, spot_price: float) -> float:
        """
        Compute the current value of the option combination.

        Args:
        - time_to_maturity (float): Time to expiration (in years).
        - spot_price (float): Current spot price.

        Returns:
        - float: Current value of the option combination.
        """
        intrinsic_val = sum([option.current_value(time_to_maturity, spot_price) for option in self.options])
        
        return intrinsic_val - self.option_initial_value
    
    def pnl_at_expiry(self, spot_at_expiry: float) -> float:
        """
        Compute the PnL at expiry for the option combination.

        Args:
        - spot_at_expiry (float): Spot price at the time of option expiry.

        Returns:
        - float: PnL at expiry.
        """
        total_pnl = sum([option.pnl_at_expiry(spot_at_expiry) for option in self.options])
        return total_pnl

    def break_even_prices(self) -> tuple:
        """
        Calculates and returns the break-even prices for the options in the combination.

        The method is currently implemented for 'straddle' and 'strangle' strategies. For both these 
        strategies, the function calculates the break-even prices as follows:

        For 'straddle':
        - The break-even price for the call option is calculated as the strike price plus the initial premium.
        - The break-even price for the put option is calculated as the strike price minus the initial premium.

        For 'strangle':
        - The break-even price for the call option (which has a higher strike price) is calculated as the 
        strike price plus the initial premium.
        - The break-even price for the put option (which has a lower strike price) is calculated as the strike 
        price minus the initial premium.

        The function returns a tuple containing the break-even prices for the call and put options, respectively.

        Returns:
            tuple: Tuple containing the break-even prices for the call and put options.

        Raises:
            NotImplementedError: If the strategy is not 'straddle' or 'strangle'.
        """

        if (self.combination == "straddle") or (self.combination == 'strangle'):
            call_put_strikes = self.get_option_strikes()
            call_put_premium = self.get_option_values()

            call_break_even = call_put_strikes[0] + call_put_premium[0]
            put_break_even = call_put_strikes[1] - call_put_premium[0]
            return (call_break_even, put_break_even)
        
        elif (self.combination == "butterfly") or (self.combination == 'condor'):
            strikes = self.get_option_strikes()
            total_premium = sum(self.get_option_values())

            lower_break_even = strikes[0] + total_premium
            upper_break_even = strikes[-1] - total_premium
            return (lower_break_even, upper_break_even)
        else:
            raise NotImplementedError(f"Break-even price calculation is not implemented for {self.combination}")



class Straddle(OptionCombination):
    """
    Represents a Straddle option strategy, a combination of a call and a put option 
    with the same strike price and expiration date.

    Inherits from OptionCombination.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, long_short: str):
        """
        Initializes a new instance of the Straddle class.

        Args:
            S (float): Current stock price.
            K (float): Strike price for both call and put options.
            T (float): Time to expiration (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            long_short (str): Whether the options are "long" or "short".
        """
        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
        assert sigma >= 0, 'Volatility (sigma) must be a non-negative value'
        assert r >= 0, 'Risk-free rate (r) must be a non-negative value'

        call_option = Option(S, K, T, r, sigma, "call", long_short)
        put_option = Option(S, K, T, r, sigma, "put", long_short)

        super().__init__([call_option, put_option])
        self.combination = 'straddle'


class Strangle(OptionCombination):
    """
    Represents a Strangle option strategy, which involves a call and a put option 
    with the same expiration date but different strike prices.

    Inherits from OptionCombination.
    """
    def __init__(self, S: float, K_low: float, K_high: float, T: float, r: float, sigma: float, long_short: str):
        """
        Initializes a new instance of the Strangle class.

        Args:
            S (float): Current stock price.
            K_low (float): Strike price for the put option.
            K_high (float): Strike price for the call option.
            T (float): Time to expiration (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            long_short (str): Whether the options are "long" or "short".
        """
        assert K_low < K_high, "Strike prices should satisfy K_low < K_high"
        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
        assert sigma >= 0, 'Volatility (sigma) must be a non-negative value'
        assert r >= 0, 'Risk-free rate (r) must be a non-negative value'

        call_option = Option(S, K_high, T, r, sigma, "call", long_short)
        put_option = Option(S, K_low, T, r, sigma, "put", long_short)

        super().__init__([call_option, put_option])
        self.combination = 'strangle'


class Butterfly(OptionCombination):
    """
    Initializes a new instance of the Strangle class.

    Args:
        S (float): Current stock price.
        K_low (float): Strike price for the put option.
        K_high (float): Strike price for the call option.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
    """


    
    def __init__(self, S: float, K_low: float, K_mid: float, K_high: float, T: float, r: float, sigma: float, long_short: str):
        """
        Initializes a new instance of the Butterfly class.

        Args:
            S (float): Current stock price.
            K_low (float): Lower strike price.
            K_mid (float): Middle strike price.
            K_high (float): Upper strike price.
            T (float): Time to expiration (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            long_short (str): Whether the options are "long" or "short".
        """
        assert K_low < K_mid < K_high, "Strike prices should satisfy K_low < K_mid < K_high"
        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
        assert sigma >= 0, 'Volatility (sigma) must be a non-negative value'
        assert r >= 0, 'Risk-free rate (r) must be a non-negative value'

        opposite_long_short = OptionCombination._opposite_long_short(long_short)

        lower_strike_call = Option(S, K_low, T, r, sigma, "call", long_short)
        middle_strike_call_1 = Option(S, K_mid, T, r, sigma, "call", opposite_long_short)
        middle_strike_call_2 = Option(S, K_mid, T, r, sigma, "call", opposite_long_short)
        upper_strike_call = Option(S, K_high, T, r, sigma, "call", long_short)

        super().__init__([lower_strike_call, middle_strike_call_1, middle_strike_call_2, upper_strike_call])
        self.combination = 'butterfly'


class Condor(OptionCombination):
    """
    Represents a Condor Spread option strategy.

    A Condor spread is constructed by holding a long and short position in two 
    pairs of different strike prices. Typically, it involves buying one lower 
    strike call, selling one lower-middle strike call, selling one upper-middle 
    strike call, and buying one upper strike call. This results in a net credit 
    to the trader's account. The strategy aims to profit from low volatility in 
    the underlying asset.

    Inherits from OptionCombination.
    """
    def __init__(self, S: float, K1: float, K2: float, K3: float, K4: float, T: float, r: float, sigma: float, long_short: str):
        """
        Initializes a new instance of the Condor class.

        Args:
            S (float): Current stock price.
            K1 (float): Strike price for the lowest call option.
            K2 (float): Strike price for the lower-middle call option.
            K3 (float): Strike price for the upper-middle call option.
            K4 (float): Strike price for the highest call option.
            T (float): Time to expiration (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            long_short (str): Whether the options are "long" or "short".
        """
        assert K1 < K2 < K3 < K4, "Strike prices should satisfy K1 < K2 < K3 < K4"
        assert long_short in ['long', 'short'], 'long_short must be either "long" or "short"'
        assert sigma >= 0, 'Volatility (sigma) must be a non-negative value'
        assert r >= 0, 'Risk-free rate (r) must be a non-negative value'

        opposite_long_short = OptionCombination._opposite_long_short(long_short)

        call1 = Option(S, K1, T, r, sigma, "call", long_short)
        call2 = Option(S, K2, T, r, sigma, "call", opposite_long_short)
        call3 = Option(S, K3, T, r, sigma, "call", opposite_long_short)
        call4 = Option(S, K4, T, r, sigma, "call", long_short)

        super().__init__([call1, call2, call3, call4])
        self.combination = 'condor'