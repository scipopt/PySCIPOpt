#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Define exceptions."""
import signal

class TimeoutError(Exception):
	"""Basic Error class for timeouts."""

	pass

class timeout:
	"""Small environment to impose a maximum time for commands."""

	def __init__(self, seconds=1, error_message='Timeout'):
		"""Setup the timer with the given amount of seconds."""
		self.seconds = seconds
		self.error_message = error_message
	def handle_timeout(self, signum, frame):
		"""Event that happens, if the maximal time is reached."""
		raise TimeoutError(self.error_message)
	def __enter__(self):
		"""Start the timer, when entering the environment."""
		signal.signal(signal.SIGALRM, self.handle_timeout)
		signal.alarm(self.seconds)
	def __exit__(self, type, value, traceback):
		"""Stop the timer, when leaving the environment."""
		signal.alarm(0)
	
class InfeasibleError(Exception):
	"""Exception to mark infeasible problems."""

	pass

class WrongModelError(Exception):
	"""Exception to mark wrong use of class."""

	pass

class NotImplementedError(Exception):
	"""Exception to mark option, that are not implemented."""

	pass
