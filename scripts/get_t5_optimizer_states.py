import pickle
import sys

import tensorflow as tf


def main(ckpt_path, output_path):
    vars = tf.train.list_variables(ckpt_path)
    states = {}
    for name, _ in vars:
        if "_slot_" not in name.split("/")[-1]:
            continue

        param_name, slot, state_type = name.rsplit("_", 2)
        assert slot == "slot"
        if param_name not in states:
            states[param_name] = {}
        assert state_type not in states[param_name]

        state = tf.train.load_variable(ckpt_path, name)
        states[param_name][state_type] = state

    pickle.dump(states, open(output_path, "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
