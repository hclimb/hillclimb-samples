import jax
import jax.numpy as jnp

def test_softmax():
    x = jnp.array([-jnp.inf, -jnp.inf])
    print(f"Softmax of all -inf: {jax.nn.softmax(x)}")

if __name__ == "__main__":
    test_softmax()
