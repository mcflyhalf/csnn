#Init file to make this a python package
import logging

def get_logger(loggerName):
	log = logging.getLogger(loggerName)
	# File handler which logs even debug messages
	fh = logging.FileHandler('csnn.log')
	fh.setLevel(logging.DEBUG)
	# Console handler that logs warnings or higher
	ch = logging.StreamHandler()
	ch.setLevel(logging.WARNING)
	# create formatter and add it to the handlers
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	# add the handlers to the logger
	print ("Adding a set of handlers now")
	log.addHandler(fh)
	log.addHandler(ch)
	log.setLevel(logging.DEBUG)
	return log