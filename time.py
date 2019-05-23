import datetime
from builtins import str


def timestamp(high_res=False):
    now = datetime.datetime.now()
    clock = [now.year, now.month, now.day, now.minute]
    if high_res:
        clock.extend([now.second, now.microsecond])
    return '_'.join([str(i) for i in clock])