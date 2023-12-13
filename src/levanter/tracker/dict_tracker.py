from .tracker import Tracker

class DictTracker(Tracker):
    """
    This is an internal class to avoid using callbacks inside train_step. Instead, auxiliary metrics are tracked
    into a dictionary, which can be logged outside of train_step.
    """

