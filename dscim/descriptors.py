# from https://github.com/dssg/dickens/blob/master/src/descriptors.py


class cachedproperty(object):
    """Non-data descriptor decorator implementing a read-only property
    which overrides itself on the instance with an entry in the
    instance's data dictionary, caching the result of the decorated
    property method.
    """

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, _type=None):
        if instance is None:
            return self

        setattr(instance, self.__func__.__name__, self.__func__(instance))

        # This descriptor is now overridden for this instance:
        return getattr(instance, self.__func__.__name__)
