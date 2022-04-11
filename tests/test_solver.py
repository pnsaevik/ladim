from ladim import timestepper
import numpy as np


class Test_Solver_run:
    def test_executes_modules_in_specified_order_until_stop(self):
        class MockParticles:
            def __init__(self, m):
                self.modules = m
                self.variables = dict(x=[])

            def __getitem__(self, item):
                return self.variables[item]

            def __repr__(self):
                return str(self.variables)

        class MockSource:
            def __init__(self, m):
                self.modules = m

            def update(self):
                self.modules['state']['x'].append(0)

        class MockOutput:
            def __init__(self, m):
                self.modules = m
                self.record = dict(pid=[], x=[])

            def update(self):
                x = self.modules['state']['x']
                self.record['x'].append(x.copy())
                self.record['pid'].append(list(range(len(x))))

        class MockTransport:
            def __init__(self, m):
                self.modules = m

            def update(self):
                x = self.modules['state']['x']
                x[:] = np.array(x) + 1

        modules = dict()
        modules['output'] = MockOutput(modules)
        modules['tracker'] = MockTransport(modules)
        modules['release'] = MockSource(modules)
        modules['state'] = MockParticles(modules)
        modules['solver'] = timestepper.TimeStepper(
            modules,
            start="2020-01-01",
            stop="2020-01-03",
            step=60*60*24,
            order=['release', 'output', 'tracker'],
        )

        modules['solver'].run()

        assert modules['output'].record['pid'] == [[0], [0, 1], [0, 1, 2]]
        assert modules['output'].record['x'] == [[0], [1, 0], [2, 1, 0]]
