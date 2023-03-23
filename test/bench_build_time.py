import cProfile
import re
import pstats

import io

pr = cProfile.Profile()
pr.enable()

import spconv
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w') as f:
    f.write(s.getvalue())

