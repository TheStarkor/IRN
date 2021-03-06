import logging

logger = logging.getLogger("base")


def create_model(opt: dict):
    model = opt["model"]

    if model == "IRN":
        from .IRN_model import IRNModel as M
    elif model == "IRN+":
        # TODO : later
        pass
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
