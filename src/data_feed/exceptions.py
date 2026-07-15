class DataFeedError(Exception):
    pass

class APIError(DataFeedError):
    pass

class ValidationError(DataFeedError):
    pass

class RateLimitError(APIError):
    pass

class CacheError(DataFeedError):
    pass

    