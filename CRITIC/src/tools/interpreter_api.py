import func_timeout


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                an = locals_.get('answer', None)
            else:
                an = [locals_.get(k, None) for k in keys]
            return an, "Done"
        except BaseException as e: # jump wrong case
            return None, repr(e)

    try:
        an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = None
        report = "TimeoutError: execution timeout"

    return an, report


def _test_safe_excute():
    code_string_1 = """import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
an = np.array([4, 5, 6])
answer = a + c"""

    code_string_2 = """budget = 1000
food = 0.3
accommodation = 0.15
entertainment = 0.25
coursework_materials = 1 - food - accommodation - entertainment
answer = budget * coursework_materials
"""
    an, report = safe_execute(code_string_1)
    print(an, report)

    an, report = safe_execute(code_string_2)
    print(an, report)


if __name__ == "__main__":
    _test_safe_excute()