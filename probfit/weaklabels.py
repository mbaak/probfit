from typing import Union, Type, Tuple, List, Sequence, Any, Optional

import numpy as np
from iminuit.util import make_func_code, describe
from scipy import special

from .funcutil import merge_func_code
from .functor import construct_arg


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def tuplize(x: Any):
    return tuple(x if isinstance(x, (list, tuple, np.ndarray)) else [] if x is None else [x])


class Node:
    def __init__(
        self, 
        indices: Union[List, Tuple, np.ndarray], 
        states: Union[List, Tuple, np.ndarray, int],
        rank: int, 
        prefix: str ='wl'
    ):
        """

        :param indices:
        """
        # basic checks on inputs
        if len(prefix) == 0:
            raise ValueError('prefix should be a filled string')
        if rank < 0:
            raise ValueError('rank should be a positive integer')

        if isinstance(states, int) and states > 0:
            states = list(range(states))
        else:
            states = list(states)
        if len(states) == 0:
            raise ValueError('states should be a filled list')

        indices = np.array(indices)
        if len(indices) == 0 or any([idx < 0 for idx in indices]):
            raise ValueError('indices should be a filled list of indices')

        self.prefix = prefix
        self.indices = indices
        indices_str = ''.join([str(idx) for idx in indices])
        self.name = f'{prefix}{indices_str}'
        self.states = states
        self.n_states = len(states)
        self.mapping = dict(zip(states, range(self.n_states)))
        self.rank = rank
        self.shape = (self.n_states - 1, self.rank)
        self.n_independent = (self.n_states - 1) * self.rank
        self.independent_probabilities = np.ones(self.n_independent)
        self._update_conditional_probabilities()

        # Leave out last probabilities, can also handle tuples of states
        states_str = [[str(s) for s in tuplize(state)] for state in states[:-1]]
        varnames = ['X'] + [f'{self.name}_{"_".join(state)}_{y}' for state in states_str for y in range(rank)]
        self.func_code = make_func_code(varnames)
        self.func_defaults = None

    def set_probabilities(self, probabilities):
        if len(probabilities) == self.n_independent:
            self.independent_probabilities = np.array(probabilities).ravel()
        elif probabilities.shape == self.conditional_probabilities.shape:
            self.independent_probabilities = probabilities[:-1]
        self._update_conditional_probabilities()

    def _update_conditional_probabilities(self):
        """Normalize the conditional probabilities
        """
        # negative entries to zero
        cprobs = self.independent_probabilities.reshape(self.shape)
        cprobs[cprobs < 0] = 0

        # dependent probabilities are one minus the rest, and cannot be negative
        dependent_probabilities = 1. - np.sum(cprobs, axis=0)
        dependent_probabilities[dependent_probabilities < 0] = 0
        dependent_probabilities = dependent_probabilities.reshape((1, -1))

        # dependent probabilities inserted at last position of cond_probs matrix.
        cprobs = np.concatenate([cprobs, dependent_probabilities], axis=0)

        # normalize each column to one.
        pnorm = cprobs.sum(axis=0)
        if not np.all(pnorm > 0):
            raise RuntimeError('conditional probabilities cannot be normalized')
        self.conditional_probabilities = cprobs / pnorm[np.newaxis, :]

    def __call__(self, *args):
        X = args[0]
        X = np.array([X] if not isinstance(X, (list, tuple, np.ndarray)) else X)
        x = X[:, self.indices if len(self.indices) > 1 else self.indices[0]] if X.ndim > 1 else X
        x_vec = vec_translate(x, self.mapping) if x.ndim == 1 else \
            np.array([self.mapping[tuple(xi)] for xi in x])

        if len(args) == self.n_independent + 1:
            self.independent_probabilities = np.array(args[1: self.n_independent + 1])
            self._update_conditional_probabilities()
        elif len(args) == 2:
            iprobs = args[1]
            if isinstance(iprobs, (list, tuple, np.ndarray)):
                self.independent_probabilities = np.array(iprobs).ravel()
                if len(self.independent_probabilities) != self.n_independent:
                    raise ValueError(f'probabilities does not have {self.n_independent} entries')
            self._update_conditional_probabilities()

        # return selected conditional probabilities
        return self.conditional_probabilities[x_vec if x_vec.shape[-1] > 1 else x_vec[0]]


class WeakLabel(Node):
    def __init__(self, index: int, states, rank, prefix='wl'):
        if index < 0:
            raise ValueError('index should be a positive integer')
        super().__init__([index], states, rank, prefix)

    def __repr__(self):
        return f"WeakLabel(name={self.name})"


class CompoundWeakLabel(Node):
    def __init__(self, nodes: Sequence[Type[Node]], prefix='cwl'):
        if len(nodes) == 0:
            raise ValueError('weak labels should be filled list of WeakLabels')

        ranks = [node.rank for node in nodes]
        if not ranks.count(ranks[0]) == len(ranks):
            raise ValueError('weak labels should have the same rank')

        indices = np.concatenate([node.indices for node in nodes])

        def outer(A, B):
            return [tuplize(x) + tuplize(y) for y in B for x in A]

        states = [None]
        for node in nodes:
            states = outer(states, node.states)

        super().__init__(indices, states, ranks[0], prefix)

    def __repr__(self):
        return f"CompoundWeakLabel(name={self.name}, indices={self.indices})"


class TrueLabel:
    def __init__(self, rank: int, prefix: str = 'pY'):
        # basic checks on inputs
        if len(prefix) == 0:
            raise ValueError('prefix should be a filled string')
        self.prefix = prefix
        if rank < 0:
            raise ValueError('rank should be a positive integer')
        self.rank = rank

        self.independent_probabilities = np.ones(self.rank - 1)
        self._update_state_probabilities()

        # set function code for bayesian model
        varnames = [f'{self.prefix}_{y}' for y in range(self.rank - 1)]
        self.func_code = make_func_code(varnames)

    def _update_state_probabilities(self):
        """Normalize the state probabilities
        """
        # negative entries to zero
        cprobs = self.independent_probabilities
        cprobs[cprobs < 0] = 0

        # dependent probability is one minus the rest, and cannot be negative
        dependent_probability = 1. - np.sum(cprobs)
        dependent_probability = 0 if dependent_probability < 0 else dependent_probability

        # dependent probability inserted at last position of states.
        cprobs = np.concatenate([cprobs, [dependent_probability]])

        # normalize sum to 1
        pnorm = cprobs.sum()
        if pnorm <= 0:
            raise ValueError('state probabilities cannot be normalized')
        self.state_probabilities = cprobs / pnorm

    def set_probabilities(self, probabilities):
        if len(probabilities) == self.rank:
            self.independent_probabilities = np.array(probabilities)[:-1]
        elif len(probabilities) == self.rank - 1:
            self.independent_probabilities = np.array(probabilities)
        self._update_state_probabilities()

    def __call__(self, *args):
        # does not take X only parameters, b/c independent of X
        if len(args) == self.rank - 1:
            self.independent_probabilities = np.array(args[0: self.rank - 1])
            self._update_state_probabilities()
        elif len(args) == 1:
            iprobs = args[0]
            if isinstance(iprobs, (list, tuple, np.ndarray)):
                self.independent_probabilities = np.array(iprobs).ravel()
                if len(self.independent_probabilities) != self.rank - 1:
                    raise ValueError(f'independent probabilities does not have {self.rank - 1} entries')
            self._update_state_probabilities()

        # return all state probabilities
        return self.state_probabilities

    def sample(self, size=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        states = range(self.rank)
        return np.random.choice(states, size=size, p=self.state_probabilities)


class BayesianModel:
    def __init__(self, nodes: Sequence[Type[Node]], prefix: str = "pY"):
        # basic checks on inputs
        if len(prefix) == 0:
            raise ValueError('prefix should be a filled string')

        if len(nodes) == 0:
            raise ValueError('nodes attribute should be filled list of nodes')

        self.indices = np.concatenate([c.indices for c in nodes])
        u_indices = np.unique(self.indices)
        if len(self.indices) != len(u_indices):
            raise ValueError('overlapping indices between nodes. nodes are not independent.')

        ranks = [node.rank for node in nodes]
        self.rank = ranks[0]
        if not ranks.count(self.rank) == len(ranks):
            raise ValueError('nodes and true_label should all have the same rank')
        self.classes = np.array(list(range(self.rank)))
        self.nodes = nodes

        # define true label (class imbalance parameters)
        self.true_label = TrueLabel(self.rank, prefix)

        # set function code
        arg = tuple(nodes) + (self.true_label, )
        self.func_code, allpos = merge_func_code(*arg)

        # store functions & caching
        self.allf = arg  # f function
        self.allpos = allpos  # position for f arg
        self.numf = len(self.allf)
        self.arglen = self.func_code.co_argcount
        self.min_pdf_value = 1e-300

    def __repr__(self):
        return f"BayesianModel(nodes=[{', '.join([str(x) for x in self.nodes])}])"

    def __call__(self, *arg):
        return self.probability(*arg)

    def set_class_balances(self, probabilities):
        self.true_label.set_probabilities(probabilities)

    def _eval(self, *arg):
        # probabilities of the true label states
        if self.arglen != len(arg) and len(arg) != 1:
            raise ValueError('wrong number of arguments')

        # multiply probabilities all weak labels and the underlying true state
        yprobs = np.ones(self.rank)
        for i in range(self.numf):
            thispos = self.allpos[i]
            if self.arglen == len(arg):
                this_arg = construct_arg(arg, thispos)
            elif len(arg) == 1 and i != self.numf - 1:
                # first argument iS X, needed by all weak labels
                this_arg = arg
            else:  # len(arg) == 1
                # true_label, does not take X
                this_arg = ()
            tmp = self.allf[i](*this_arg)
            yprobs = yprobs * tmp

        return yprobs

    def probability(self, *arg):
        """Return probability estimates for the test vector X.

        :param X: array-like of shape (n_samples, n_features)
        :return: array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        yprobs = self._eval(*arg)

        # sum probabilities over the true label states
        probs = yprobs.sum(axis=1) if len(yprobs.shape) == 2 else yprobs.sum()

        # prevent probability values of zero. if so return tiny value instead
        is_array = isinstance(probs, np.ndarray)
        probs = np.array(probs if is_array else [probs])
        probs[probs < self.min_pdf_value] = self.min_pdf_value
        return probs if is_array else probs[0]

    def log_probability(self, *arg):
        """Return log-probability estimates for the test vector X.

        :param X: array-like of shape (n_samples, n_features)
        :return: array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.log(self.probability(*arg))

    def predict(self, *arg):
        """Perform classification on an array of test vectors X.

        :param X: array-like of shape (n_samples, n_features)
        :return: ndarray of shape (n_samples,)
            Predicted target values for X
        """
        prob_class = self.predict_proba(*arg)
        return self.classes[np.argmax(prob_class, axis=1)]

    def predict_log_proba(self, *arg):
        """Return log-probability estimates for the test vector X.

        :param X: array-like of shape (n_samples, n_features)
        :return: array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.log(self.predict_proba(*arg))

    def predict_proba(self, *arg):
        """ Return probability estimates for the test vector X.

        :param X: array-like of shape (n_samples, n_features)
        :return: array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        # sum probabilities over the true label states
        prob_class = self._eval(*arg)
        prob_sum = prob_class.sum(axis=1) if len(prob_class.shape) == 2 else prob_class.sum()

        # prevent probability values of zero. if so return tiny value instead
        is_array = isinstance(prob_sum, np.ndarray)
        sum_zero_check = np.array(prob_sum if is_array else [prob_sum])
        zero_mask = sum_zero_check < self.min_pdf_value
        if np.any(zero_mask):
            # need to redo prob_sum
            prob_class[zero_mask] = self.min_pdf_value
            prob_sum = prob_class.sum(axis=1) if len(prob_class.shape) == 2 else prob_class.sum()

        # per class probability
        proba = prob_class / (prob_sum[:, np.newaxis] if is_array else prob_sum)
        return proba

    def sample(self, size=1, seed=None, include_true_label=False):
        """
        :param size:
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        n_col = np.max(self.indices) + 1
        n_col += 1 if include_true_label else 0
        X = np.zeros(shape=(size, n_col), dtype=int)

        # generate true labels
        Y = self.true_label.sample(size=size)
        y, counts = np.unique(Y, return_counts=True, axis=0)
        if include_true_label:
            X[:, -1] = Y

        # generate weak labels, one by one.
        for node in self.nodes:
            shape = (size, len(node.indices)) if len(node.indices) > 1 else (size, )
            samples = np.empty(shape=shape, dtype=int)
            states = range(node.n_states)
            for i, size_y in enumerate(counts):
                # choice does not seems to work with states of shape 2
                s = np.random.choice(states, size=size_y, p=node.conditional_probabilities[:, y[i]])
                samples[(Y == y[i])] = np.array(node.states)[s]
            X[:, node.indices if len(node.indices) > 1 else node.indices[0]] = samples
        return X


class BinnedLabels:
    def __init__(self, X: Union[List, Tuple, np.ndarray], select_indices: Optional[Union[List, Tuple, np.ndarray]] = None):
        """"""
        # basic checks on inputs
        if len(X) == 0:
            raise ValueError('X should be a filled numpy array')
        X = np.array(X)

        if select_indices is not None:
            select_indices = np.array(select_indices)
            if len(select_indices) == 0 or any([idx < 0 for idx in select_indices]):
                raise ValueError('select_indices should be a filled list of valid indices')
            X = X[:, select_indices if len(select_indices) > 1 else select_indices[0]] if X.ndim > 1 else X

        # make binned dataset with counts per bin
        # we don't store empty bins to save memory.
        self.bin_x, self.bin_entries = np.unique(X, return_counts=True, axis=0)
        self.n_samples = np.sum(self.bin_entries)
        # dict only takes tuples as keys
        self.data = dict(zip([tuple(x) for x in self.bin_x], self.bin_entries))


class BinnedLH:
    def __init__(self, binned_data: BinnedLabels, bayes_model: BayesianModel):
        self.data = binned_data
        self.bm = bayes_model

        # skip X! that will be binned_data
        varnames = describe(bayes_model)[1:]
        self.func_code = make_func_code(varnames)
        self.arglen = len(varnames)
        self.parameter_names = varnames

    def default_errordef(self):
        return 0.5

    def __call__(self, *arg):
        if self.arglen != len(arg):
            raise ValueError('wrong number of arguments')

        x_arg = (self.data.bin_x, ) + arg
        probs = self.bm(*x_arg)

        n_observed = self.data.n_samples
        # predicted number of entries per bin, for all filled bins
        f = n_observed * probs
        # predicted number of entries for bins with zero entries
        f_zero = np.max(n_observed * (1. - np.sum(probs)), 0)

        # observed number of entries
        y = self.data.bin_entries

        # Poisson binned likelihood for filled bins
        nll = f - special.xlogy(y, f) + special.gammaln(y + 1)

        # likelihood sum of filled and zero bins
        return np.sum(nll) + f_zero

    def _test_stats(self):
        # total number of entries
        n_observed = self.data.n_samples

        # predicted number of entries per bin, for all filled bins
        x = self.data.bin_x
        probs = self.bm(x)
        f = n_observed * probs

        # observed number of entries per bin
        y = self.data.bin_entries
        return f, y

    def gtest(self):
        f, y = self._test_stats()
        g_test = 2. * y * np.log(y / f)
        return np.sum(g_test)

    def chi2(self):
        f, y = self._test_stats()
        chi2 = (y - f) ** 2 / f
        return np.sum(chi2)

    def psi(self):
        f, y = self._test_stats()
        psi = y * np.log10(y / f)
        return 10 * np.sum(psi)
