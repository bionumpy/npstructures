import numpy as np
import cProfile
import pstats
from npstructures import RaggedArray, Counter
N=5
hashes = (np.load(f"/home/knut/Sources/kmer_mapper/h{i}.npy") for i in range(N))
ragged = RaggedArray.load("profiling/.fullragged.npz")
#agged = RaggedArray.load("profiling/.new_fullragged.npz")
# ragged.save("profiling/.new_fullragged.npz")
counter = Counter(ragged, safe_mode=False)
# h = next(hashes)
# counter.count(np.insert(ragged._data, 0, 5))
p_stats_name = "profiling/.count.txt"
if False:
    for h in hashes:
        counter.count(h)
    exit()
cProfile.run("[counter.count(h) for h in hashes]", p_stats_name)
stats = pstats.Stats(p_stats_name)
# stats.sort_stats("tottime")
stats.sort_stats("cumulative")
stats.print_stats()
