import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm.auto import trange


class LatentClassConsensus(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_blocks,
        em_max_iter=50,
        adam_max_iter=300,
        n_inits=50,
        random_state=None,
        # algorithm="em",
    ):
        self.n_blocks = n_blocks
        self.adam_max_iter = adam_max_iter
        self.em_max_iter = em_max_iter
        self.n_inits = n_inits
        self.random_state = random_state

    def fit(self, X, y=None):
        self.valid_ = np.any(X >= 0, axis=1)
        rg = np.random.default_rng(self.random_state)

        self.valid_ = np.any(X >= 0, axis=1)
        print(f"{self.valid_.shape=} {self.valid_.sum()=}")
        best_loglik = -np.inf

        if self.valid_.any():
            for i in trange(self.n_inits, desc="LCA inits"):
                Q, pi, B, logliks = em(
                    self.n_blocks, X[self.valid_], max_iter=self.em_max_iter, rg=rg, init_kind="svd" if not i else None
                )
                if logliks[-1] > best_loglik:
                    self.objectives_, probs, self.pi_, self.B_ = logliks, Q, pi, B
                    print("new em best", logliks[-1], "old best", best_loglik)
                    best_loglik = logliks[-1]

        # probs, self.pi_, self.B_, self.objectives_ = em(
        #     self.n_blocks, X[self.valid_], max_iter=self.em_max_iter, rg=rg
        # )

        # best_loglik = -np.inf
        # if self.valid_.any():

        # for f in trange(self.n_inits):
        #     pi_logit_init = rg.normal(size=(self.n_blocks))
        #     B_logit_init = rg.normal(size=(X.shape[1], self.n_blocks, X.max() + 1))
        #     Q, pi, B, logliks = adam(
        #         pi_logit_init, B_logit_init, X[self.valid_], max_iter=self.adam_max_iter, init_kind="svd" if not f else None
        #     )
        #     if logliks[-1] > best_loglik:
        #         self.objectives_, probs, self.pi_, self.B_ = logliks, Q, pi, B
        #         print("new adam best", logliks[-1], "old best", best_loglik, "f", f)
        #         best_loglik = logliks[-1]

        self.probabilities_ = np.zeros((X.shape[0], self.n_blocks))
        self.probabilities_[self.valid_] = probs

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.valid_, self.probabilities_.argmax(1), -1)


def one_hot(labels):
    """NS -> NSK. Zeros for -1s so that missing values are ignored below."""
    K = labels.max() + 1
    e = np.pad(np.eye(K), ((0, 1), (0, 0)))
    labels = np.where(labels >= 0, labels, K)
    return e[labels]


def posterior(labels, pi, B, one_hots):
    N, S, K = one_hots.shape
    # T = B.shape[1]

    # B: S, T, K
    # one_hots: N, S, K
    # B_obs: N, S, T
    B_obs = np.einsum("stk,nsk->nst", B, one_hots)
    # N,S
    invalid = np.nonzero(~(one_hots > 0).any(axis=2))
    B_obs[invalid] = 1.0
    B_obs = B_obs.prod(1)
    # Q = B_obs + np.log(pi)
    # Q = torch.softmax(torch.from_numpy(Q), dim=1).numpy(force=True)
    Q = B_obs * pi
    denom = Q.sum(1)
    Q[denom > 0] /= denom[denom > 0][:, None]
    # print(f"{Q.min()=}")
    # print(f"{Q.max()=}")
    # denom = Q.sum(1, keepdims=True)
    # Q /= denom
    assert np.all(Q >= 0)
    assert np.all(Q <= 1)
    assert np.all(np.isclose(Q.sum(1), 1.0))
    assert np.isclose(Q.sum(), Q.shape[0])
    return Q


def update_pi(Q, one_hots):
    N, T = Q.shape

    # n,s
    # valid = (one_hots > 0).any(axis=2)
    # valid_n = valid.sum(axis=1)
    # pi = np.einsum("n,nt->t", valid_n, Q)
    # pi /= pi.sum()
    pi = Q.mean(0)
    # print(f"{pi.min()=} {pi.max()=}")
    assert np.isclose(pi.sum(), 1.0)
    return pi


def update_B(Q, one_hots):
    valid = (one_hots > 0).any(axis=2)
    B = np.einsum("nt,nsk->stk", Q, one_hots)
    denom = np.einsum("ns,nt->st", valid, Q)
    # B_ntsk = Q[:, :, None, None] * one_hots[:, None, :, :]
    # B_tsk = B_ntsk.sum(0)
    # B = B_tsk.transpose(1, 0, 2)
    # denom = B.sum(2)
    # denom[denom == 0] = 1
    B = B / denom[:, :, None]
    B = np.clip(B, 1e-5, 1 - 1e-5)
    B /= B.sum(2, keepdims=True)
    # print(f"{B.min()=} {B.max()=}")
    assert np.all(B >= 0)
    assert np.all(B <= 1)
    # assert np.all(np.isclose(B.sum(2), 1))
    return B


def log_likelihood(pi, B, one_hots):
    B_obs = np.einsum("stk,nsk->nst", B, one_hots)
    valid = (one_hots > 0).any(axis=2)
    B_obs[~valid] = 1.0
    B_obs = B_obs.prod(1)
    # B_obs = B_obs[valid]
    log_marginals = np.log(B_obs @ pi)
    return np.sum(log_marginals)


def elbo(pi, B, labels, one_hots):
    Q = posterior(labels, pi, B, one_hots)
    entropy = -(Q * np.log(Q)).sum()
    prior = (Q * np.log(pi)).sum()
    B_obs = np.einsum("nt,stk,nsk->ns", Q, B, one_hots)
    valid = np.nonzero((one_hots > 0).any(axis=2))
    # B_obs = B_obs[valid]
    # log_marginals = np.log(B_obs @ pi)
    likelihood = np.sum(np.log(B_obs[valid]))

    return prior + entropy + likelihood


def em(T, labels, max_iter, rg, init_kind=None):
    N, S = labels.shape
    K = labels.max() + 1

    one_hots = one_hot(labels)
    if init_kind == "svd":
        N, S, K = one_hots.shape
        u, s, vh = np.linalg.svd(one_hots.reshape(N, S * K), full_matrices=False)
        pi = torch.softmax(torch.from_numpy((s**2)[:T]), dim=0).numpy(force=True)
        B = torch.softmax(torch.from_numpy(vh[:T].reshape(T, S, K).transpose(1, 0, 2)), dim=2).numpy(force=True)
        pi = np.clip(pi, 1e-5, 1 - 1e-5)
        pi /= pi.sum()
        B = np.clip(B, 1e-5, 1 - 1e-5)
        B /= B.sum(2, keepdims=True)
    else:
        Q = np.exp(rg.normal(size=(N, T)))
        Q /= Q.sum(1)[:, None]
        # pi, B = m_step(Q, one_hots)
        pi = update_pi(Q, one_hots)
        B = update_B(Q, one_hots)
    # print(f"{init_kind=} {one_hots.shape=}")

    logliks = [log_likelihood(pi, B, one_hots)]
    for i in (range if max_iter < 100 else trange)(max_iter):
        Q = posterior(labels, pi, B, one_hots)
        ll = log_likelihood(pi, B, one_hots)
        # print(f"{i=} {ll=}")
        assert ll >= logliks[-1], f"0 {i=} {ll=} {logliks[-1]=}"
        B = update_B(Q, one_hots)
        # pi = update_pi(Q, one_hots)
        ll2 = log_likelihood(pi, B, one_hots)
        # print(f"{i=} {ll2=}")
        assert ll2 + 1e-2 >= ll, f"1 {i=} {ll2=} {ll=}"
        Q = posterior(labels, pi, B, one_hots)
        pi = update_pi(Q, one_hots)
        # B = update_B(Q, one_hots)
        ll3 = log_likelihood(pi, B, one_hots)
        # print(f"{i=} {ll3=}")
        assert ll3 + 1e-2 >= ll2, f"2 {i=} {ll3=} {ll2=}"

        logliks.append(log_likelihood(pi, B, one_hots))
        assert logliks[-1] + 1e-2 >= logliks[-2], f"{logliks[-1]=} {ll=} {logliks[-2]=}"

    Q = posterior(labels, pi, B, one_hots)

    return Q, pi, B, logliks


def dirichlet_categorical_pmf(alpha, dim=1):
    # sum_alpha = alpha.sum(dim=dim)
    lgamma_2 = torch.lgamma(torch.tensor(2.0, dtype=alpha.dtype))
    # logZ = torch.lgamma(sum_alpha) - torch.lgamma(1.0 + sum_alpha)
    kernels = torch.lgamma(1.0 + alpha) - torch.lgamma(alpha) - lgamma_2
    return torch.softmax(kernels, dim=dim)


def loglik(pi, B, one_hots):
    B = torch.clip(B, 1e-5, 1 - 1e-5)
    pi = torch.clip(pi, 1e-5, 1 - 1e-5)
    B_obs = torch.einsum("stk,nsk->nst", B, one_hots)
    valid = torch.nonzero((one_hots > 0).any(dim=2), as_tuple=True)
    # print(f"{B.shape=} {B.shape[1]=}")
    # print(f"{one_hots.shape=}")
    # print(f"{B_obs.shape=}")
    # print(f"{valid.shape=}")
    # print(f"{B_obs.min()=} {B_obs.max()=} {B_obs.shape=}")
    B_obs = B_obs[valid]
    # print(f"{B_obs.shape=}")
    log_marginals = torch.log(B_obs @ pi)
    # print(f"{log_marginals.min()=} {log_marginals.max()=} {log_marginals.shape=}")
    # log_marginals = log_marginals.view(-1)[valid]
    # print(f"{log_marginals.min()=} {log_marginals.max()=} {log_marginals.shape=}")
    return torch.sum(log_marginals)


def elbo_(pi, B, labels, one_hots):
    Q = posterior(labels, pi, B, one_hots)
    entropy = -(Q * np.log(Q)).sum()
    prior = (Q * np.log(pi)).sum()
    B_obs = np.einsum("nt,stk,nsk->ns", Q, B, one_hots)
    valid = np.nonzero((one_hots > 0).any(axis=2))
    # B_obs = B_obs[valid]
    # log_marginals = np.log(B_obs @ pi)
    likelihood = np.sum(np.log(B_obs[valid]))

    return prior + entropy + likelihood


def adam(pi_logit_init, B_logit_init, labels, max_iter=300, init_kind=None, distribution="cat"):
    one_hots = torch.from_numpy(one_hot(labels))
    T = pi_logit_init.size
    if init_kind == "svd":
        N, S, K = one_hots.shape
        u, s, vh = np.linalg.svd(one_hots.reshape(N, S * K), full_matrices=False)
        pi_logit = torch.log_softmax(torch.from_numpy(s[:T]), dim=0)
        B_logit = torch.log_softmax(torch.from_numpy(vh[:T].reshape(T, S, K).transpose(1, 0, 2)), dim=2)
    else:
        pi_logit = torch.from_numpy(pi_logit_init)
        B_logit = torch.from_numpy(B_logit_init)
    pi_logit.requires_grad_()
    B_logit.requires_grad_()

    if distribution == "cat":
        normalizer = F.softmax
    elif distribution == "dircat":
        normalizer = dirichlet_categorical_pmf
    else:
        assert False

    opt = torch.optim.Adam([pi_logit, B_logit], lr=0.01)
    logliks = []
    for i in range(max_iter):
        opt.zero_grad()
        # Q_logit.requires_grad_()
        # grad, loss = grad_and_value(adam_elbo)(Q_logit, masks, Delta)
        # Q_logit.grad = grad
        # Q_logit -= 0.01 * grad
        loss = -loglik(
            F.softmax(pi_logit, dim=0),
            normalizer(B_logit, dim=2),
            one_hots,
        )
        loss.backward()
        # if not i:
        #     print(torch.abs(pi_logit.grad).max())
        opt.step()
        logliks.append(loss.numpy(force=True))

        # if i == 100:
        #     for g in opt.param_groups:
        #         g['lr'] = 0.01

        # if i == 100:
        #     for g in opt.param_groups:
        #         g['lr'] = 0.1

    pi = F.softmax(pi_logit, dim=0).numpy(force=True)
    B = normalizer(B_logit, dim=2).numpy(force=True)
    Q = posterior(labels, pi, B, one_hots)
    return Q, pi, B, -np.array(logliks)
