"""
Parameter container for XRD image processing configuration.
"""
from dataclasses import dataclass

@dataclass
class ProcessingParams:
    """
    Parameters for cosmic ray removal processing.
    
    All parameters are stored as strings to simplify UI integration,
    but are converted to appropriate types when used.
    
    Default values:
    - th_donut (15): Threshold for donut detection
    - th_mask (0.05): Threshold for mask creation (0-1)
    - th_streak (3): Threshold for streak detection
    - win_streak (3): Window size for streak detection
    - exp_donut (9): Expansion of donut mask
    - exp_streak (3): Expansion of streak mask
    """
    # Default values are set here as class attributes
    th_donut: str = '15'
    th_mask: str = '0.05'
    th_streak: str = '3'
    win_streak: str = '3'
    exp_donut: str = '9'
    exp_streak: str = '3'

    def validate(self) -> None:
        """
        Validate all parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Convert and validate th_donut
        try:
            th_donut = int(self.th_donut)
            if th_donut < 0:
                raise ValueError('Threshold of donut should be a non-negative integer')
        except ValueError as e:
            if 'invalid literal for int()' in str(e):
                raise ValueError('Threshold of donut should be an integer') from e
            raise

        # Convert and validate th_mask
        try:
            th_mask = float(self.th_mask)
            if not 0 <= th_mask <= 1:
                raise ValueError('Threshold of mask should be a float between 0 and 1')
        except ValueError as e:
            if 'could not convert string to float' in str(e):
                raise ValueError('Threshold of mask should be a float') from e
            raise

        # Convert and validate th_streak
        try:
            th_streak = int(self.th_streak)
            if th_streak < 0:
                raise ValueError('Threshold of streak should be a non-negative integer')
        except ValueError as e:
            if 'invalid literal for int()' in str(e):
                raise ValueError('Threshold of streak should be an integer') from e
            raise

        # Convert and validate win_streak
        try:
            win_streak = int(self.win_streak)
            if win_streak < 0:
                raise ValueError('Window size of streak should be a non-negative integer')
        except ValueError as e:
            if 'invalid literal for int()' in str(e):
                raise ValueError('Window size of streak should be an integer') from e
            raise

        # Convert and validate exp_donut
        try:
            exp_donut = int(self.exp_donut)
            if exp_donut < 0:
                raise ValueError('Expansion of donut should be a non-negative integer')
        except ValueError as e:
            if 'invalid literal for int()' in str(e):
                raise ValueError('Expansion of donut mask should be an integer') from e
            raise

        # Convert and validate exp_streak
        try:
            exp_streak = int(self.exp_streak)
            if exp_streak < 0:
                raise ValueError('Expansion of streak should be a non-negative integer')
        except ValueError as e:
            if 'invalid literal for int()' in str(e):
                raise ValueError('Expansion of streak mask should be an integer') from e
            raise

        # Additional cross-parameter validation
        if th_streak > win_streak * win_streak:
            raise ValueError('Threshold of streak should be less than window size squared')