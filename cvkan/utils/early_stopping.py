import torch
class EarlyMinStopper():
    """
        If EarlyStopper sees no improvement larger than 'threshold' for 'patience' steps, it will return .stop_training() = True
        :param patience: How many steps to tolerate no improvement for
        :param threshold: How big the change needs to be at least to be counted as improvement
    """
    def __init__(self, patience, threshold, logging_interval=50):
        self.patience = patience
        self.threshold = threshold
        self.steps_without_improvement = 0
        self.best_value = torch.tensor(float("inf"))
        self.stop_criterion_reached = False
        self.logging_interval = logging_interval
    def step(self, value):
        if value < self.best_value - self.threshold:
            # Improvement
            self.best_value = value
            self.steps_without_improvement = 0
        else:
            # No Improvement
            self.steps_without_improvement += 1
            if self.logging_interval > 0 and self.steps_without_improvement % self.logging_interval == 0:
                print(f"EarlyStopper noticed no improvement in current epoch! Steps without Improvement: {self.steps_without_improvement} / {self.patience}")
        if self.steps_without_improvement > self.patience:
            self.stop_criterion_reached = True

    def should_stop(self):
        return self.stop_criterion_reached

