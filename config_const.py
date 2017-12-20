import logging

# DEBUG Detailed information, typically of interest only when diagnosing problems.
# INFO  Confirmation that things are working as expected.
# WARNING   An indication that something unexpected happened.
# ERROR Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL  A serious error, indicating that the program itself may be unable to continue running.
FORMAT = "[%(asctime)5s - %(filename)10s: - line # %(lineno)5s - %(funcName)12s() ] %(message)s"
log_level = logging.DEBUG
