from time import time
from typing import TypeVar, ParamSpec, Callable

def print_time(text : str, print_verbose : bool = False):
    """Decorator to print time necessary for the computation
    of the function. The time will be printed along the
    given `text`
    
    Note by default no verbosity, needs to be set to true somewhere

    Args:
        text (str): Text to be printed with time
    """
    T = TypeVar("T")
    P = ParamSpec("P")
    def decorator(f : Callable[P, T]) -> Callable[P, T]:
        def _inner(*args: P.args, **kwargs: P.kwargs) -> T:
            
            verbose = print_verbose

            if not len(args) == 0 and hasattr(args[0], 'verbose') and args[0].verbose:
                verbose = True
            elif 'verbose' in kwargs.keys():
                verbose = True

            if verbose:
                print(f'Start {text}.', end=" ", flush=True)

            start = time()
            
            # execute acual function call
            result = f(*args, **kwargs)

            elapsed = time() - start
            elapsed = elapsed
            if verbose:
                print(f'Finised {text} in {elapsed} seconds.')
            return result
        return _inner
    return decorator

class FunctionTimer():
    
    def __init__(self, verbose : bool = False):
        """Class to provide timer functionality
        
        To be in herited by other classes.

        Args:
            verbose (bool, optional): Time will be printed for decorated function if set to True. Defaults to False.
        """
        self.verbose = verbose
