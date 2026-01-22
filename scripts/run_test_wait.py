import os
import sys
import time
import subprocess


def main():
	# Short default sleep to allow quick attach during manual runs.
	# For long sleeps (debugging), set environment variable `DM_LONG_SLEEP=1`.
	long_sleep = os.environ.get('DM_LONG_SLEEP', '') in ('1', 'true', 'True')
	print('PID=' + str(os.getpid()))
	print('Sleeping 5s to allow debugger/procdump attach...')
	sys.stdout.flush()
	time.sleep(5)
	if long_sleep:
		print('Long sleep enabled; sleeping 300s')
		sys.stdout.flush()
		time.sleep(300)

	# run pytest
	args = [sys.executable, '-u', '-m', 'pytest', 'tests/ai/test_batch_inference.py::test_batch_inference_basic', '-q', '-s']
	print('Running:', ' '.join(args))
	sys.stdout.flush()
	ret = subprocess.call(args)
	print('pytest exit', ret)


if __name__ == '__main__':
	main()
