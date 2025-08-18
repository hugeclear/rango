#!/usr/bin/env python3
import os, time
base = "runs"
candidates = []
if os.path.isdir(base):
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            candidates.append((os.path.getmtime(p), p))
if not candidates:
    print("")
else:
    print(sorted(candidates, key=lambda x: x[0], reverse=True)[0][1])
