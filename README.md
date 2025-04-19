A similar data-structure to a [VecDeque](VecDeque), in that it is cyclic, but instead of cycling around its capacity, it cycles around its length.

In short, this makes rotation and SISO-behaviour simpler, but all operations that change the delay-line's length require it to be made contiguous first.