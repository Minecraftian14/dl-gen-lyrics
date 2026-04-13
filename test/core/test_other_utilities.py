import os, random, re

from generator_core import key_cached, cached


def test_key_cached():
    for test_file in ["test_key.bone", "test_key.pkl"]:
        file = os.path.join("temp", test_file)
        if os.path.exists(file):  os.remove(file)

    function_calls = [0]

    def function():
        function_calls[0] += 1
        return 1114

    key_cached("test_key", function)
    assert function_calls[0] == 1
    key_cached("test_key", function)
    assert function_calls[0] == 1


class Something:
    @cached()
    def thing(self):
        return random.random()


def test_cached():
    for test_file in ["Something.thing.cached.bone", "Something.thing.cached.pkl"]:
        file = os.path.join("temp", test_file)
        if os.path.exists(file): os.remove(file)

    something = Something()
    thing = something.thing()
    assert something.thing() == thing
