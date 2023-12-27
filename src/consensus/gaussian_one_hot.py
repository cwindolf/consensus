import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm.auto import trange


class GaussianOneHotConsensus(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_blocks,
        max_iter=200,
        n_inits=1,
        random_state=None,
    ):
        self.n_blocks = n_blocks
        self.max_iter = max_iter
        self.n_inits = n_inits
        self.random_state = random_state

    def fit(self, X, y=None):
        best_elbo = -np.inf
        rg = np.random.default_rng(self.random_state)
        self.valid_ = np.any(X >= 0, axis=1)
        probs_ = None
        if self.valid_.any():
            for f in trange(self.n_inits, desc="Gauss inits"):
                elbos, probs, pi, P, B = adam_fit(
                    X[self.valid_], T=self.n_blocks, max_iter=self.max_iter, rg=rg
                )
                if elbos[-1] > best_elbo:
                    self.objectives_, probs_, self.pi_, self.P_, self.B_ = (
                        elbos,
                        probs,
                        pi,
                        P,
                        B,
                    )
                    print("new best", elbos[-1], "old best", best_elbo, "f", f)
                    best_elbo = elbos[-1]
        self.probabilities_ = np.zeros((X.shape[0], self.n_blocks))
        self.probabilities_[self.valid_] = probs_

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.valid_, self.probabilities_.argmax(1), -1)


def make_delta(labels):
    N, S = labels.shape
    masks = np.zeros((S, N, N))
    Delta = np.zeros((S, N, N))
    for s, l in enumerate(labels.T):
        valid = np.flatnonzero(l >= 0)
        masks[s, valid[:, None], valid[None, :]] = 1
        my_labels = np.unique(l)
        for u in my_labels[my_labels >= 0]:
            in_u = np.flatnonzero(l == u)
            Delta[s, in_u[:, None], in_u[None, :]] = 1
    masks[:, np.arange(N), np.arange(N)] = 0
    return masks, Delta


def elbo(Q_logit, P_logit, B_logit, one_hots, masks):
    S, T, K = B_logit.shape
    N = Q_logit.shape[0]

    Q = F.softmax(Q_logit, dim=1)
    logQ = F.log_softmax(Q_logit, dim=1)
    P = F.sigmoid(P_logit)
    # logP = F.logsigmoid(P_logit)
    # log1P = F.logsigmoid(P_logit.neg())
    B = F.softmax(B_logit, dim=2)
    pi = Q.mean(0)

    # mD = masks * Delta
    # m1D = masks * (1 - Delta)

    # -- prior
    prior = (Q * torch.log(pi)).sum()

    # -- entropy
    entropy = -(Q * logQ).sum()

    # -- likelihood
    # offidag positive
    Lambdas = torch.einsum(
        "stk,sul,it,ju,tu,sij->sikjl",
        B,
        B,
        Q,
        Q,
        P,
        masks,
    )
    Lambdas = Lambdas.view(S, N * K, N * K)
    diag_Lambdas = Lambdas.sum(2)
    # offdiag negative
    Lambdas = Lambdas.neg()
    Lambdas = torch.diagonal_scatter(Lambdas, 1 + diag_Lambdas, dim1=1, dim2=2)
    mus = torch.einsum("it,stk->sik", Q, B)
    dmus = one_hots.permute(1, 0, 2) - mus
    dmus = dmus.reshape(S, N * K)
    mahal = torch.einsum("sp,spq,sq->", dmus, Lambdas, dmus)
    # svs = torch.linalg.svdvals(Lambdas)
    # logdet = torch.sum(torch.log(svs))
    logdet = torch.logdet(Lambdas)
    # print(f"{svs.sum()=} {(svs > 1e-5).sum(1).min()=} {(svs > 1e-5).sum(1).max()=}")
    likelihood = -0.5 * mahal - 0.5 * logdet.sum()

    # print(f"{prior=} {entropy=} {likelihood=}")
    # print(f"{mahal=} {logdet=}")

    return -(prior + entropy + likelihood)


def one_hot(labels):
    """NS -> NSK. Zeros for -1s so that missing values are ignored below."""
    K = labels.max() + 1
    e = np.pad(np.eye(K), ((0, 1), (0, 0)))
    labels = np.where(labels >= 0, labels, K)
    return e[labels]


def adam_fit(labels, T, max_iter, rg, init_kind="svd"):
    N, S = labels.shape
    K = labels.max() + 1

    masks, Deltas = make_delta(labels)
    masks = torch.from_numpy(masks)
    one_hots = one_hot(labels)
    one_hots = torch.from_numpy(one_hots)

    if init_kind == "svd":
        N, S, K = one_hots.shape
        u, s, vh = np.linalg.svd(one_hots.reshape(N, S * K), full_matrices=False)
        B_logit = torch.log_softmax(
            torch.from_numpy(vh[:T].reshape(T, S, K).transpose(1, 0, 2)), dim=2
        )
        Q_logit = torch.from_numpy(u[:, :T])
        Q_logit = torch.log_softmax(Q_logit, dim=1)
        P_logit = rg.normal(size=(T, T))
        P_logit = P_logit @ P_logit.T
    else:
        Q_logit = rg.normal(size=(N, T))
        P_logit = rg.normal(size=(T, T))
        P_logit = P_logit @ P_logit.T
        B_logit = rg.normal(size=(S, T, K))

    Q_logit = torch.as_tensor(Q_logit).detach().clone()
    Q_logit.requires_grad_()
    P_logit = torch.as_tensor(P_logit).detach().clone()
    P_logit.requires_grad_()
    B_logit = torch.as_tensor(B_logit).detach().clone()
    B_logit.requires_grad_()

    opt = torch.optim.Adam([Q_logit, P_logit, B_logit], lr=0.1)
    elbos = []
    for i in trange(max_iter, desc="fit"):
        opt.zero_grad()
        # Q_logit.requires_grad_()
        # grad, loss = grad_and_value(adam_elbo)(Q_logit, masks, Delta)
        # Q_logit.grad = grad
        # Q_logit -= 0.01 * grad
        loss = elbo(Q_logit, P_logit, B_logit, one_hots, masks)
        loss.backward()
        # print(f"{torch.abs(Q_logit.grad).max()=} {loss=}")
        opt.step()
        elbos.append(loss.numpy(force=True))

        # if not i % 10:
        dq = torch.abs(Q_logit.grad).max()
        if dq < 1e-4:
            # print(f"break {i=} {dq=}")
            break

        # if i == 20:
        #     for g in opt.param_groups:
        #         g['lr'] = 0.1

    Q = torch.softmax(Q_logit, dim=1).numpy(force=True)
    P = torch.sigmoid(P_logit).numpy(force=True)
    pi = Q.mean(0)
    B = torch.softmax(B_logit, dim=2).numpy(force=True)
    return -np.array(elbos), Q, pi, P, B
