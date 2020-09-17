class ProcessedData:

    def __init__(
        self,
        train,
        val,
        test,
        train_labels,
        val_labels,
        test_labels,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
