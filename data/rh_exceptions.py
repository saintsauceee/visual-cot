class RushHourException(Exception):
    """Base exception class for Rush Hour errors."""
    pass

class InvalidMove(RushHourException):
    """Raised when an invalid move is attempted."""
    pass

class CarNotFound(RushHourException):
    """Raised when a specified car is not found on the board."""
    pass

