import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_random_state, check_array, _get_n_jobs

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

    def _get_prior(self, prior, t):
        if isinstance(prior, float):
            return prior
        elif isinstance(prior, list):
            return prior[t]
        elif callable(prior):
            return prior(t)

    def _init_latent_vars(self, X):
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

        self.word_topic_ = {}
        self.doc_topic_count_ = np.zeros([X.shape[0], self.n_components])
        self.topic_word_count_ = {}
        self.topic_count_ = np.zeros([self.n_components])

        topic_distrib = self.random_state_.dirichlet(np.ones(self.n_components) *
                                                     self._get_prior(self.doc_topic_prior_, self.n_iter_), X.shape[0])

        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                if X[i][j] != 0:
                    topic = np.random.choice(self.n_components, 1, p=topic_distrib[i])[0]
                    self.word_topic_[(i, j)] = topic
                    self.doc_topic_count_[i][topic] += 1
                    self.topic_word_count_[(topic, j)] = self.topic_word_count_.get((topic, j), 0) + 1
                    self.topic_count_[topic] += 1

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
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every

        # initialize parameters
        self._init_latent_vars(X)
        # change to perplexity later
        last_bound = None
        for i in range(max_iter):
            for (doc, word) in self.word_topic_:
                old_topic = self.word_topic_[(doc, word)]
                self.doc_topic_count_[doc][old_topic] -= 1
                self.topic_word_count_[(old_topic, word)] -= 1
                self.topic_count_[old_topic] -= 1
                new_topic = self.gibbs_sample()
                self.doc_topic_count_[doc][new_topic] += 1
                self.topic_word_count_[(new_topic, word)] += 1
                self.topic_count_[new_topic] += 1

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

    def score(self, X, y=None):
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        X = self._check_non_neg_array(X, "LatentDirichletAllocation.score")

        doc_topic_distr = self._unnormalized_transform(X)
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def _perplexity_precomp_distr(self, X, doc_topic_distr=None,
                                  sub_sampling=False):
        """Calculate approximate perplexity for data X with ability to accept
        precomputed doc_topic_distr

        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : array-like or sparse matrix, [n_samples, n_features]
            Document word matrix.

        doc_topic_distr : None or array, shape=(n_samples, n_components)
            Document topic distribution.
            If it is None, it will be generated by applying transform on X.

        Returns
        -------
        score : float
            Perplexity score.
        """
        if not hasattr(self, 'components_'):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("no 'components_' attribute in model."
                                 " Please fit model first.")

        X = self._check_non_neg_array(X,
                                      "LatentDirichletAllocation.perplexity")

        if doc_topic_distr is None:
            doc_topic_distr = self._unnormalized_transform(X)
        else:
            n_samples, n_components = doc_topic_distr.shape
            if n_samples != X.shape[0]:
                raise ValueError("Number of samples in X and doc_topic_distr"
                                 " do not match.")

            if n_components != self._n_components:
                raise ValueError("Number of topics does not match.")

        current_samples = X.shape[0]
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)

        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt

        return np.exp(-1.0 * perword_bound)

    def perplexity(self, X, doc_topic_distr='deprecated', sub_sampling=False):
        """Calculate approximate perplexity for data X.

        Perplexity is defined as exp(-1. * log-likelihood per word)

        .. versionchanged:: 0.19
           *doc_topic_distr* argument has been deprecated and is ignored
           because user no longer has access to unnormalized distribution

        Parameters
        ----------
        X : array-like or sparse matrix, [n_samples, n_features]
            Document word matrix.

        doc_topic_distr : None or array, shape=(n_samples, n_components)
            Document topic distribution.
            This argument is deprecated and is currently being ignored.

            .. deprecated:: 0.19

        Returns
        -------
        score : float
            Perplexity score.
        """
        if doc_topic_distr != 'deprecated':
            warnings.warn("Argument 'doc_topic_distr' is deprecated and is "
                          "being ignored as of 0.19. Support for this "
                          "argument will be removed in 0.21.",
                          DeprecationWarning)

        return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)

    def _unnormalized_transform(self, X):
        """Transform data X according to fitted model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_components)
            Document topic distribution for X.
        """
        if not hasattr(self, 'components_'):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("no 'components_' attribute in model."
                                 " Please fit model first.")

        # make sure feature size is the same in fitted model and in X
        X = self._check_non_neg_array(X, "LatentDirichletAllocation.transform")

        doc_topic_distr, _ = self._e_step(X, cal_sstats=False,
                                          random_init=False)

        return doc_topic_distr

    def transform(self, X):
        """Transform data X according to the fitted model.

           .. versionchanged:: 0.18
              *doc_topic_distr* is now normalized

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_components)
            Document topic distribution for X.
        """
        doc_topic_distr = self._unnormalized_transform(X)
        doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr
