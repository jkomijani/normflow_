# Copyright (c) 2021-2022 Javad Komijani

"""A module for modifying torch.sort by introducing a new order."""


import torch
import numpy as np


# =============================================================================
class Order:
    """A class to perform troch.sort, but with a method to revert the sorting.

    Creating an instance and sorting the input are integrated.

    Once an instance is created, the argsort indices (the indices that sort
    a tensor along a given dimension in ascending order by value) are obtained
    and saved along with the sorted values (as properties `sorted_ind` and
    `sorted_val`, respectively).

    Use the `sort` method to sort a new tensor with respect to `sorted_ind`.
    Use the `revert` method to revert the sorting operation.
    """
    def __init__(self, x, dim=-1):
        self.dim = dim
        self.sorted_val, self.sorted_ind = self._sort(x, dim=dim)

    def _sort(self, x, dim=-1):
        return x.sort(dim=dim)

    def sort(self, x):
        """Sort (gather) x according to self.sorted_ind (if already defined)."""
        return x.gather(self.dim, self.sorted_ind)

    def revert(self, x):
        """Revert the sorting (gathering) operation."""
        return x.gather(self.dim, torch.argsort(self.sorted_ind))


# =============================================================================
class ZeroSumOrder(Order):
    """A class to perform troch.sort but after imposing the zero sum condition.

    For details of initiating, sorting, and reverting see `Order`.
    For details of imposing zero sum condition see `self._sort`.
    """
    def _sort(self, x, dim=-1):
        """Given a tensor of phases `x`, sort the tensor after a "designated"
        element is changed to make the sum of tensor elemenets zero
        (in the `dim` dimension).

        Idially, it is assumed that the input phases `x` are in `(-pi, pi]`,
        and their sum over the axis specified by `dim` is a multiplicative of
        `2 pi`.
        No warning will be raised if the ideal assumption is not satisfied.

        We use "canonic" to denote a specific transformation of input phases.
        First, the input phases are ordered; this gives inital ordered indices.
        Second, depending on the sum of input phases on the axis specified by
        `dim`, one phase is designated to be shifted by a factor of `2 pi` so
        that the total sum vanishes. Then all phases are in `(-pi, pi]` except
        at most one "designated" phase.
        Third, after shifting the value of "designated" phase, the previously
        sorted values are sorted again.

        The method returns the final zero-sum-sorted values and indices.

        See `zerosum` method for explanation of the method of specifying the
        "designated" phase.
        """
        val, ind = x.sort(dim=dim)
        val = self.zerosum(val, dim=dim)
        val, ind_prime = val.sort(dim=dim)  # "val" needs to be sorted agian
        ind = ind.gather(dim, ind_prime)  # "ind" must be sorted accordingly
        return val, ind

    @staticmethod
    def zerosum(x, dim=-1, period=2 * np.pi):
        """Change a "designated" element such that the sum of `x` vanishes.

        The designated element is obtained by first finding the windig number
        of the input `x`, i.e., the sum modula the period of elements.
        If he winding number is the minimum (maximum) possible value, the first
        (last) element is the designated one, otherwise a simple linear map
        specifies the designate value.
        """
        n_w = roundint(torch.sum(x, dim=dim) / period).unsqueeze(dim)
        n_d = x.shape[-1]
        # we now determine the designated item and change its value to the
        # zero-sum end. The index of designated item is (n_w + n_d//2) % n_d
        x.scatter_add_(dim, (n_w + n_d//2) % n_d, -period * n_w)
        return x


# =============================================================================
class ModalOrder:
    """A class to sort eigenvalues of a matrix based on directions of
    corresponding eigenvectors.

    Once an instance is created, the argsort indices are calculated and saved
    as `sorted_ind`.

    Use the `sort` method to sort the eigenvalues or the matrix of egienvectors
    with respect to `sorted_ind`.
    Use the `revert` method to revert the sorting operation.
    """
    def __init__(self, modal_matrix, **sort_kwargs):
        """
        Parameter
        ---------
        modal_matrix : tensor
            The matrix of eigenvectors
        """
        self.dim = -1  # hard-wired
        self.sorted_ind = self.modal_argsort(modal_matrix, **sort_kwargs)

    @property
    def sorted_ind_4matrix(self):
        """Return adjusted sorted_ind such that one can sort matrices."""
        ind = self.sorted_ind
        return ind.unsqueeze(-1).repeat(*[1]*len(ind.shape), ind.shape[-1]).transpose(-1, -2)


    def sort(self, x):
        """Sort (gather) x according to self.sorted_ind (if already defined)."""
        ind = self.sorted_ind
        if len(x.shape) > len(ind.shape):  # then x is a matrix
            ind = self.sorted_ind_4matrix
        return x.gather(self.dim, ind)

    def revert(self, x):
        """Revert the sorting (gathering) operation."""
        ind = self.sorted_ind
        if len(x.shape) > len(ind.shape):  # then x is a matrix
            ind = self.sorted_ind_4matrix
        return x.gather(self.dim, torch.argsort(ind))

    @staticmethod
    def modal_argsort(modal_matrix, row=None):
        """Return (argsort) indices for sorting the columns of modal_matrix.

        Note that modal_matrix is supposed to be square (in the last two axes).

        Parameters
        ----------
        matrix : Tensor
            A batch of square matrices

        row: None or int (optional, default=None)
            A simple algorithm for sorting the columns based on a row of the
            matrix.
            (Works well for SU(n) matrices matrices because the elements of the
            rows of SU(n) matrices are statiscally never zero or equal, and
            therefore one can easily sort them. This simple method is four
            times faster the main method for SU(3) matrices.)
        """
        dim = -1  # hard-wired

        if isinstance(row, int): # argsort the `row` row of modal_matrix
            return torch.argsort(torch.abs(modal_matrix[..., row, :]), dim=dim)

        # Main method:
        # 1. find arguments to sort (fromm smallest to largest item)
        argsort = torch.argsort(torch.abs(modal_matrix), dim=dim)

        # 2. Take the last column of argsort as a starting point
        argsort_col = torch.select(argsort, dim, -1)
        # argsort_col can have repeated indices that should be removed as below

        arange_like_ = arange_like(argsort_col)

        ndim = argsort.shape[dim - 1]
        ind = torch.ones_like(argsort_col) * (ndim - 1)
        ind_select = torch.arange(ndim)

        # 3. find the repeated indices in argsort_col, and change them
        for _ in range(ndim - 1):
            for j in range(ndim - 1):
                arg_j = argsort_col.select(-1, j).unsqueeze(-1)
                cond = (argsort_col - arg_j == 0) & (arange_like_ > j)
                ind[cond] -= 1
                if torch.sum(cond):  # correct `argsort_col` for repetitive indices
                    argsort_col = torch.gather(argsort, -1, ind.unsqueeze(-1)).squeeze(-1)

        return argsort_col

    @staticmethod
    def _modal_argsort_sanitycheck(modal_matrix, **kwargs):
        """As a sanity check, e.g. use:

           > prior = normflow.prior.SUnPrior(n=3)
           > mat = prior.sample(10000)
           > ModalOrder = normflow.lib.matrix_handles.ordering.ModalOrder
           > ModalOrder._modal_argsort_sanitycheck(mat, row=None)
           ... tensor(0)
           > ModalOrder._modal_argsort_sanitycheck(mat, row=0)
           ... tensor(0)

           > %timeit ModalOrder(mat, row=None)
           2.42 ms ± 94 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

           > %timeit ModalOrder(mat, row=0)
           605 µs ± 27 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        """
        arg = ModalOrder.modal_argsort(modal_matrix, **kwargs)
        return torch.sum(torch.abs(torch.sort(arg, dim=-1)[0] - arange_like(arg, dim=-1)))


# =============================================================================
class _OBSOLETE_SUnPhaseOrder:

    def __init__(self, x, dim=-1, **kwargs):
        self.sorted_ind = self.rolled_argsort(x, dim=dim, **kwargs)
        self.sort_axis = dim

    def __call__(self, x):
        return x.gather(self.sort_axis, self.sorted_ind)

    def revert(self, x):
        return x.gather(self.sort_axis, torch.argsort(self.sorted_ind))

    @staticmethod
    def rolled_argsort(phases, canonic_roll=True, dim=-1):
        """Identical to `torch.argsort` if `canonic_roll=False`. Otherwise,
        depending on the sum of phases, returns a rolled version of the output
        of `torch.argsort`, where the shift is determined by the index of the
        "designated" phase as explained below.

        Ignore the rest if `canonic_roll=False`.

        Idially, it is assumed that the input `phases` are in `(-pi, pi]`, and
        their sum on the axis specified by `dim` is a multiplacative of `2 pi`.
        No warning will be raised if the ideal assumption is not satisfied.

        We use "canonic" to denote a specific transformation of input phases.
        First, the input phases are ordered; this gives inital ordered indices.
        Second, depending on the sum of input phases on the axis specified by
        `dim`, one phase is designated to be imaginarily shifted by a factor of
        `2 pi` such that the sum vanishes.
        (Should the imaginary shift performed, all phases are in `(-pi, pi]`
        except at most one "designated" phase.)
        Third, the inital ordered indices are rolled such that the "designated"
        phase is moved to the end of the list.
        The method returns the final rolled indices.

        We now explain the method of specifying the "designated" phase....
        """
        sorted_ind = phases.argsort(dim=dim)
        if not canonic_roll:
            return sorted_ind
        n_ph = phases.shape[dim]
        winding = roundint(torch.sum(phases, dim=dim).unsqueeze(dim) / (2 * np.pi))
        canonic_ind = (arange_like(phases, dim=dim) + winding - n_ph // 2) % n_ph
        return sorted_ind.gather(dim, canonic_ind)


# =============================================================================
def arange_like(x, dim=-1):
    # copied from lib/arange.py
    """Return a tensor with shape of `x`, filled with `(0, 1, ..., n)` in the
    `dim` direction, where `n = x.shape[dim]`, and repeated in all other
    directions.
    """
    if dim == -1 or dim == x.ndim - 1:
        # a special case, which can be written in a compact form as
        return torch.arange(x.shape[-1]).repeat((*x.shape[:-1], 1))

    arange = torch.arange(x.shape[dim])
    subshape = x.shape[1+dim:]
    arange = arange.view(-1, *[1]*len(subshape)).repeat((1, *subshape))
    if dim % x.ndim > 0:
        arange = arange.repeat((*x.shape[:dim], *[1]*arange.ndim))
    return arange


# =============================================================================
def roundint(x, dtype=torch.int64):
    """Return the closest integer to `x`."""
    x_ =  torch.round(x)
    return x_ if dtype is None else x_.type(dtype)
