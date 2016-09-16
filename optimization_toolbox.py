import csv, numpy, decimal, re, os, sys, inspect
import numpy as np
from numpy import NaN, inf
import scipy.interpolate
from pint import UnitRegistry
import matplotlib.pyplot as plt
import unittest
import json
from functools import partial, wraps
import cma
import optimization_toolbox_common as otc

class _T_Stupid_Identity(object): 
    @staticmethod
    def eval(x): return x
    @staticmethod
    def eval_inv(x): return x

class T_BaseClass(object):
    def __init__(self, inner=None, invert=False):
        self.inner = inner if inner else _T_Stupid_Identity
        if invert:
            self.f, self.g = self.g, self.f
    def eval(self, x):
        return self.f(self.inner.eval(x))
    def eval_inv(self, x):
        return self.inner.eval_inv(self.g(x))
    @staticmethod
    def f(x): return x
    @staticmethod
    def g(x): return x

T_Identity_Instance = T_BaseClass()

class T_Log(T_BaseClass):
    def __init__(self, inner=None, **kw):
        super().__init__(inner, **kw)
        self.f = np.log
        self.g = np.exp

class T_PintUnit(T_BaseClass):
    def __init__(self, inner=None, unit=None):
        assert unit is not None
        self.u = unit
        self.f = lambda x:(x/u).to('1')
        self.g = lambda x:x*u
        super().__init__(inner)

class T_Linear(T_BaseClass):
    def __init__(self, inner=None, x0=None, y0=None, x1=None, y1=None, a=None, b=None, **kw):
        if a is None:
            if x1 is not None:
                a = (y1-y0)/(x1-x0)
            else:
                a = 1.0
        if b is None:
            if x0 is not None:
                b = y0 - x0*a
            else:
                b = 0.0
        self.a = a
        self.b = b
        self.f = lambda x: x*a+b
        self.g = lambda x: (x-b)/a
        super().__init__(inner, **kw)

class ParameterTransform(object):
    '''do not modify any properties after initialization!'''
    dependent_parameters = frozenset()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ParameterTransform()
    def init_ParameterTransform(self):
        parameters = self.parameters
        self.num_parameters = len(parameters)
        dep = self.dependent_parameters
        self.active_parameters = [x for x in parameters if x[0] not in dep]
        self.num_active_parameters = len(self.active_parameters)
    def compute_dependent(self, kw):
        '''update kw to add missing dependent parameters self.dependent_parameters'''
        return
    def transform(self, kw):
        return [transform.eval(kw[name]) for name,transform in self.active_parameters]
    def untransform(self, xs, dependent=True):
        mp = self.active_parameters
        assert len(mp)==len(xs)
        kw = dict((name,transform.eval_inv(a)) for a,(name,transform) in zip(xs,mp))
        if dependent:
            self.compute_dependent(kw)
        return kw

class TestPT(unittest.TestCase):
    def test_pt(self):
        class MyTransform(ParameterTransform):
            parameters = [('a', T_Log()),
                          ('b', T_Identity_Instance)]
            dependent_parameters = frozenset(['b'])
            def compute_dependent(self, kw):
                kw.update(b=4.0)
        
        pt = MyTransform()
        
        self.assertEquals(pt.transform(dict(a=1)), [0.0])
        self.assertEquals(pt.untransform([0.0], dependent=False), {'a':1.0})
        self.assertEquals(pt.untransform([0.0]), {'a':1.0, 'b':4.0})
        
    def test_pt_composition_inversion(self):
        class MyTransform(ParameterTransform):
            parameters = [('a', T_Linear(T_Log(), a=2, b=1, invert=True))]
        pt = MyTransform()
        self.assertEquals(pt.untransform([-0.5])['a'], 1.0)
        self.assertEquals(pt.transform(dict(a=1.0))[0], -0.5)

class TransformedSolution(object):
    def __init__(self, problem, solver, parameter_vector, value, *args, **kwargs):
        self.problem = problem
        self.solver  = solver
        self.value   = value
        self.parameter_vector = parameter_vector
        self.parameter_dict = problem.untransform(parameter_vector)
        super().__init__(*args, **kwargs)

class TransformedProblem(ParameterTransform):
    '''common class for optimization and curve fitting problems'''
    solution_class = TransformedSolution
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        call = self.__call__
        utr = self.untransform
        self.call_transformed = lambda a: call(utr(a))
        
        if not hasattr(self, 'penalty_scale'):
            self.penalty_scale = 1e3
        if not hasattr(self, 'constraints_scale'):
            self.constraints_scale = 1.0
        else:
            typical_p = self.untransform(next(self.p0()), dependent=True)
            self.constraints_scale = np.array([
                (typical_p[x] if isinstance(x, str) else x)
                for x in self.constraints_scale])
    def constraints(self, kw):
        '''this method should return a numpy array with component v[i] of order
self.constraints_scale[i], where a constraint is violated iff a component is
negative.'''
        return np.array([])
    def check_constraints(self, kw):
        '''returns None if no constraints are violated'''
        v = self.constraints(kw)
        b = v < 0.0
        if any(b):
            #return np.nan
            #print('crap', kw)
            return (sum((np.where(b,v,0.0)/self.constraints_scale)**2)+sum(b))*self.penalty_scale
        return None
    def __call__(self, kw):
        '''this method should call and use self.check_constraints()'''
        raise NotImplementedError()
    def p0(self):
        """return iterator of initial parameter sets; must be already transformed."""
        return iter([[0.5]*self.num_active_parameters])
    
class OutOfBounds(Exception):
    def __init__(self, penalty):
        self.penalty = penalty

class CurveFitSolution(TransformedSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.problem.model(self.parameter_dict)

class CurveFitProblem(TransformedProblem):
    '''you must define model(self, kw) which must return function f(x) given parameters in kw'''
    solution_class = CurveFitSolution
    def __init__(self, *args, Xs, Ys, **kwargs):
        self.Xs = Xs
        self.Ys = Ys
        if not hasattr(self, 'penalty_scale'):
            self.penalty_scale = sum(Ys**2) * 1e3
        super().__init__(*args, **kwargs)
    def __call__(self, kw):
        '''this is the objective function. feel free to override.'''
        v = self.check_constraints(kw)
        if v is not None: return v
        f = self.model(kw)
        return np.sum((f(self.Xs) - self.Ys)**2)

class DiffEqCurveFitProblem(CurveFitProblem):
    '''you must define self.de_solver(self, kw), which must return Ys corresponding to self.Xs'''
    def __call__(self, kw):
        v = self.check_constraints(kw)
        if v is not None:
            #print('bad',kw)
            return v
        FYs = self.de_solver(kw)
        return np.sum((FYs - self.Ys)**2)
    def model(self, kw):
        return scipy.interpolate.InterpolatedUnivariateSpline(
            self.Xs, self.de_solver(kw))

class Minimizer(object):
    solution = None
    always_replace = False
    @property
    def solution_class(self):
        return self.problem.solution_class
    def __init__(self, *, problem, **kwargs):
        self.problem = problem
        super().__init__(**kwargs)
    def set_solution(self, *, parameter_vector, value, **kw):
        sol = getattr(self, 'solution', None) if self.always_replace else None
        if sol is None or sol.value >= value:
            self.solution = self.solution_class(
                problem=self.problem,
                solver=self,
                parameter_vector=parameter_vector,
                value=value)
    def run(self):
        '''must return self!'''
        raise NotImplementedError()
        return self

class ComposingMinimizer(Minimizer):
    def __init__(self, *, minimizers=[], **kwargs):
        self.minimizers = [m(problem=kwargs['problem'])
                           for m in minimizers]
        super().__init__(**kwargs)
    def set_solution(self, **kw):
        raise NotImplementedError("you shouldn't use this method in ComposingMinimizer")
    def run(self):
        for m in self.minimizers:
            m.run()
        self.update()
    def update(self):
        sol = self.solution
        for m in self.minimizers:
            m_sol = m.solution
            if m_sol is not None and (sol is None or sol.value >= m_sol.value):
                sol = m_sol
        self.solution = sol
        self.notify_global_solution()
    def notify_global_solution(self):
        sol = self.solution
        for m in self.minimizers:
            # notify other minimizers of global solution
            method = getattr(m, 'notify_global_solution', None)
            if method is not None:
                method(sol)

class SavingMinimizer(Minimizer):
    do_save = True
    def run(self):
        try:
            with open(self.filename) as f:
                sdict = json.load(f)
        except FileNotFoundError:
            pass
        else:
            svect = self.problem.transform(sdict)
            val = self.problem.call_transformed(svect)
            self.set_solution(parameter_vector=svect,
                              value=val)
    def notify_global_solution(self, solution):
        if self.do_save and solution is not None:
            self.set_solution(parameter_vector=solution.parameter_vector,
                              value=solution.value)
            solution = self.solution
            sdict = self.problem.untransform(solution.parameter_vector)
            otc.atomic_json_write(self.filename, sdict)

class ConvenientSavingMinimizerFactory(object):
    def filename(self, name):
        return 'MinimizerResult_{}.json'.format(name)
    def __call__(self, name, problem, solver, *, run_solver=True, use_saved=True):
        ms = []
        if run_solver:
            ms.append(solver)
        fn = self.filename(name)
        class MySavingMinimizer(SavingMinimizer):
            filename = fn
            def run(self):
                if use_saved:
                    super().run()
        ms.append(MySavingMinimizer)
        minimizer = ComposingMinimizer(problem=problem, minimizers=ms)
        return minimizer

class CurveFitter(Minimizer):
    solution_class = CurveFitSolution

class NullCMALogger(cma.BaseDataLogger):
    modulo = 100000
    def _blackhole(*args,**kwargs): pass
    __init__ = add = register = disp = plot = data = _blackhole

class CMAMinimizer(Minimizer):
    def __init__(self, *, sigma0, inopts={}, null_logger=True, **kwargs):
        problem = kwargs['problem']
        self.es = es = cma.CMAEvolutionStrategy(x0=next(problem.p0()), sigma0=sigma0, inopts=inopts)
        if null_logger:
            es.logger = NullCMALogger()
        super().__init__(**kwargs)
    def run(self):
        es = self.es
        res = es.optimize(self.problem.call_transformed).result()
        self.set_solution(parameter_vector=res[0], value=res[1])
        return self

class CMACurveFitter(CurveFitter, CMAMinimizer): pass

class TestCMAMinimizer(unittest.TestCase):
    def test_cma_min(self):
        class RosenbrockProblem(TransformedProblem):
            parameters = [('x{}'.format(i), T_Identity_Instance) for i in range(5)]
            def __call__(self, kw):
                xs = np.array([x[1] for x in sorted(kw.items())])
                return cma.fcts.rosen(xs)
        
        m = CMAMinimizer(problem=RosenbrockProblem(),
                         sigma0=1.0)
        m.run()
        self.assertTrue(m.solution.value < 1e-6)
    
    def test_cma_min_with_constraints(self):
        class MyProblem(TransformedProblem):
            parameters = [('x', T_Identity_Instance),
                          ('y', T_Identity_Instance),
                          ('R', T_Identity_Instance)]
            dependent_parameters = frozenset(['R'])
            def compute_dependent(self, kw):
                kw.update(R=1.0)
            penalty_scale = 1e6
            def __call__(self, kw):
                v = self.check_constraints(kw)
                if v is not None: return v
                return -(kw['x'] + kw['y'])
            def constraints(self, kw):
                return np.array([kw['R']**2 - (kw['x']**2 + kw['y']**2)])
        
        m = CMAMinimizer(problem=MyProblem(),
                         sigma0=2.0)
        m.run()
        s = m.solution.parameter_dict
        self.assertAlmostEqual(s['x'], 0.7071067811865476, 4)
        self.assertAlmostEqual(s['y'], 0.7071067811865476, 4)
        
    def test_cma_fit(self):
        Xs = np.array([1,2,3,4,5,6,7])
        Ys = np.array([1.1,2.2,3.3,4.2,5.8,6.4,7.2]) + 10
        
        for mapping in [T_Identity_Instance, T_Log(), T_Linear(T_Log(), a=3)]:
            class LinearFitProblem(CurveFitProblem):
                parameters = [('a', mapping), ('b', T_Identity_Instance)]
                def model(self, kw):
                    return lambda x: x*kw['a'] + kw['b']
            m = CMACurveFitter(problem=LinearFitProblem(Xs=Xs, Ys=Ys), sigma0=0.6)
            m.run()
            s = m.solution.parameter_dict
            self.assertAlmostEqual(s['a'], 1.04285, 4)
            self.assertAlmostEqual(s['b'], 10.14285, 4)
            self.assertAlmostEqual(m.solution.model(4), 14.31, 1) # evaluate model function given solution parameters
            #print(m.solution.parameter_dict, m.solution.parameter_vector)



