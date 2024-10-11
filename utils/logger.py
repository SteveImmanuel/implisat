import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def get_logger(name: str, is_disabled: bool = False):
    # adapted from https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                """Accept every signature by doing non-operation."""
                pass
            return no_op

    if not is_disabled:
        return logging.getLogger(name)
    return NoOp()