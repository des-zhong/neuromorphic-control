import lyngor as lyn
import numpy as np
import time

if __name__ == "__main__":
    sample = np.random.rand(11).astype(np.float32)
    r = lyn.load(path='./Net_0/', device=0)
    t1 = time.time()
    while True:
        r.run(input=sample)
    print('ppppppppppppppppppppppppppppppp:',time.time()-t1)
