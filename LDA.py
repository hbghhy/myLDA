
import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_random_state,check_array,_get_n_jobs

from sklearn.utils.validation import check_non_negative
import warnings


class LDA:
    """Latent Dirichlet Allocation with online variational Bayes algorithm

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

    Parameters
    ----------
    n_components : int, optional (default=10)
        Number of topics.

    doc_topic_prior : float, list, function of iteration optional (default=None)
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
        In the literature, this is called `alpha`.

    topic_word_prior : float, list, function of iteration optional (default=None)
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
        In the literature, this is called `eta`.


    max_iter : integer, optional (default=10)
        The maximum number of iterations.


    evaluate_every : int optional (default=0)
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evalute perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    perp_tol : float, optional (default=1e-1)
        Perplexity tolerance in batch learning. Only used when
        ``evaluate_every`` is greater than 0.

    max_sample_iter : int (default=100)
        Max number of iterations for sample topic of word.

    n_jobs : int, optional (default=1)
        The number of jobs to use in the E-step. If -1, all CPUs are used. For
        ``n_jobs`` below -1, (n_cpus + 1 + n_jobs) are used.

    verbose : int, optional (default=0)
        Verbosity level.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    components_ : array, [n_components, n_words]
        Distribution over the words for each topic:
        ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

    n_batch_iter_ : int
        Number of iterations of the EM step.

    n_iter_ : int
        Number of passes over the dataset.

    References
    ----------
    [1] "Online Learning for Latent Dirichlet Allocation", Matthew D. Hoffman,
        David M. Blei, Francis Bach, 2010

    [2] "Stochastic Variational Inference", Matthew D. Hoffman, David M. Blei,
        Chong Wang, John Paisley, 2013

    [3] Matthew D. Hoffman's onlineldavb code. Link:
        http://matthewdhoffman.com//code/onlineldavb.tar

    """

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, max_iter=10,
                 evaluate_every=-1,
                 perp_tol=1e-1, max_sample_iter=100,
                 n_jobs=1, verbose=0, random_state=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_iter = max_iter
        self.evaluate_every = evaluate_every
        self.perp_tol = perp_tol
        self.max_sample_iter = max_sample_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state


    def _check_params(self):
        """Check model parameters."""
        self._n_components = self.n_components

        if self._n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self._n_components)

    def _init_latent_vars(self,n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self._n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self._n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self._n_components, n_features))



    def _check_non_neg_array(self, X, whom):
        """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
        X = check_array(X, accept_sparse='csr')
        check_non_negative(X, whom)
        return X


    def fit(self, X, y=None):
        """Learn model for the data X with variational Bayes method.

        When `learning_method` is 'online', use mini-batch update.
        Otherwise, use batch update.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        self
        """
        self._check_params()
        X = self._check_non_neg_array(X, "LDA.fit")
        n_samples, n_features = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method
        if learning_method is None:
            warnings.warn("The default value for 'learning_method' will be "
                          "changed from 'online' to 'batch' in the release "
                          "0.20. This warning was introduced in 0.18.",
                          DeprecationWarning)
            learning_method = 'online'

        batch_size = self.batch_size

        # initialize parameters
        self._init_latent_vars(n_features)
        # change to perplexity later
        last_bound = None
        n_jobs = _get_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0,
                      self.verbose - 1)) as parallel:
            for i in xrange(max_iter):
                if learning_method == 'online':
                    for idx_slice in gen_batches(n_samples, batch_size):
                        self._em_step(X[idx_slice, :], total_samples=n_samples,
                                      batch_update=False, parallel=parallel)
                else:
                    # batch update
                    self._em_step(X, total_samples=n_samples,
                                  batch_update=True, parallel=parallel)

                # check perplexity
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    doc_topics_distr, _ = self._e_step(X, cal_sstats=False,
                                                       random_init=False,
                                                       parallel=parallel)
                    bound = self._perplexity_precomp_distr(X, doc_topics_distr,
                                                           sub_sampling=False)
                    if self.verbose:
                        print('iteration: %d of max_iter: %d, perplexity: %.4f'
                              % (i + 1, max_iter, bound))

                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound

                elif self.verbose:
                    print('iteration: %d of max_iter: %d' % (i + 1, max_iter))
                self.n_iter_ += 1

        # calculate final perplexity value on train set
        doc_topics_distr, _ = self._e_step(X, cal_sstats=False,
                                           random_init=False,
                                           parallel=parallel)
        self.bound_ = self._perplexity_precomp_distr(X, doc_topics_distr,
                                                     sub_sampling=False)

        return self
