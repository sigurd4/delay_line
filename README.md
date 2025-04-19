A delay-line buffer for real-time use.

# Examples

In this example, we mix in a delayed version of the signal `x`, delayed by 2 samples.

```rust
use delay_line::*;

let mut x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

let mut delay = delay_line![0.0; 2];

for x in &mut x
{
    *x += delay.delay(*x)*0.5;
}

assert_eq!(x, [1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
```