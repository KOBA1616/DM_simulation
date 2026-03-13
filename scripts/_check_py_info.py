import sys, platform, struct
print(sys.version)
print(sys.executable)
print(platform.architecture())
print(struct.calcsize('P')*8)
