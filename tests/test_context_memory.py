import unittest
from tools.context_memory import cpp_chunks, document_chunks, Scheduler, State, fold_event

class ContextTests(unittest.TestCase):
    def test_cpp(self):
        chunks = cpp_chunks('a.cpp', '#include <vector>\nstruct Config { int limit; int run() { return limit; } };')
        self.assertTrue(any('Config: int run()' in c.skeleton for c in chunks))
        self.assertTrue(any('int limit;' == c.skeleton for c in chunks))
        with self.assertRaises(ValueError): cpp_chunks('bad.cpp', 'int f( {')

    def test_budget_and_swap(self):
        chunks = document_chunks('report.md', '# Risk\nCapital ratio 8.2%.\n\n# Other\nUnrelated facts.\n')
        # Bytes are an explicit test budget, not a model-token estimate.
        scheduler = Scheduler(chunks, lambda s: len(s.encode()))
        ids = scheduler.route('capital ratio')
        self.assertEqual(len(ids), 1)
        plan = scheduler.assemble('rules', 'index', 'capital?', '', ids, 1000, 20)
        self.assertIn('8.2%', plan['prompt'])
        self.assertLessEqual(plan['tokens'] + 20, 1000)
        next_plan = scheduler.assemble('rules', 'index', 'next?', '', [], 1000)
        self.assertNotIn('8.2%', next_plan['prompt'])
        omitted = scheduler.assemble('rules', 'index', 'capital?', '', ids, 70)
        self.assertEqual(omitted['omitted'], ids)
        with self.assertRaises(ValueError):
            scheduler.assemble('rules', 'index' * 100, 'q', '', [], 20)
        with self.assertRaises(KeyError):
            scheduler.assemble('', '', '', '', ['missing'], 1000)

    def test_state(self):
        state = State()
        state.write('database', 'postgres', 'turn1')
        with self.assertRaises(ValueError): state.write('database', 'mongo', 'turn2')
        state.write('database', 'mongo', 'turn2', expected=('postgres', 'turn1'))
        with self.assertRaises(ValueError): state.write('goal', 'x', '')
        self.assertIn('[ref=step3]', fold_event(3, 'search', '404', 'not\nfound', 'step3'))

if __name__ == '__main__': unittest.main()
