import logging
import time

log_file = "./logfile.log"
logging.basicConfig(filename=log_file,
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)


# all in one
class Log(object):
    def __init__(self, name=__name__):
        self.__name = name
        self.__logger = logging.getLogger(self.__name)

    @property
    def Logger(self):
        return self.__logger