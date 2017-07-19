from Zoo.Prelude import *

__all__ = [ "update_target_graph" ]

def update_target_graph(from_scope, to_scope, tau=1.0):
    """
    These functions allows us to update the parameters of our target network
    with those of the self.primary_network network.
    """
    assert tau <= 1.0 and tau > 0.0

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope.name)
    to_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope.name)
    updated_q = lambda tvar, fvar: (tau * fvar.value()) + ((1 - tau) * tvar.value())
    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(updated_q(to_var, from_var)))

    return op_holder


