from ladim import solver
import numpy as np


class Test_Solver_run:
    def test_executes_modules_in_specified_order_until_stop(self):
        class MockParticles:
            def __init__(self):
                self.variables = dict(x=[])

            def __getitem__(self, item):
                return self.variables[item]

            def __repr__(self):
                return str(self.variables)

        class MockSource:
            # noinspection PyMethodMayBeStatic
            def update(self, m):
                m.modules['state']['x'].append(0)

        class MockOutput:
            def __init__(self):
                self.record = dict(pid=[], x=[])

            def update(self, m):
                x = m.modules['state']['x']
                self.record['x'].append(x.copy())
                self.record['pid'].append(list(range(len(x))))

        class MockTransport:
            # noinspection PyMethodMayBeStatic
            def update(self, m):
                x = m.modules['state']['x']
                x[:] = np.array(x) + 1

        class MockModel:
            def __init__(self, m):
                self.modules = m

            def __getattr__(self, item):
                return self.modules[item]

        modules = dict()
        modules['output'] = MockOutput()
        modules['tracker'] = MockTransport()
        modules['release'] = MockSource()
        modules['state'] = MockParticles()
        modules['solver'] = solver.Solver(
            start="2020-01-01",
            stop="2020-01-03",
            step=60*60*24,
            order=['release', 'output', 'tracker'],
        )

        model = MockModel(modules)
        modules['solver'].run(model)

        assert modules['output'].record['pid'] == [[0], [0, 1], [0, 1, 2]]
        assert modules['output'].record['x'] == [[0], [1, 0], [2, 1, 0]]
