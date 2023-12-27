import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm.auto import trange


class BernoulliEdgeConsensus(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_blocks,
        max_iter=300,
        n_inits=50,
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
            for f in trange(self.n_inits, desc="IndEdges inits"):
                elbos, probs, pi, P, B = adam_fit(
                    X[self.valid_], T=self.n_blocks, max_iter=self.max_iter, rg=rg
                )
                if elbos[-1] > best_elbo:
                    self.objectives_, probs_, self.pi_, self.P_, self.B_ = elbos, probs, pi, P, B
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


def elbo(Q_logit, P_logit, B_logit, masks, Delta):
    Q = F.softmax(Q_logit, dim=1)
    logQ = F.log_softmax(Q_logit, dim=1)
    P = F.sigmoid(P_logit)
    logP = F.logsigmoid(P_logit)
    # log1P = F.logsigmoid(P_logit.neg())
    B = F.softmax(B_logit, dim=2)
    pi = Q.mean(0)

    mD = masks * Delta
    m1D = masks * (1 - Delta)

    # -- prior
    prior = (Q * torch.log(pi)).sum()

    # -- entropy
    entropy = -(Q * logQ).sum()

    # -- likelihood
    BB = torch.einsum("stk,suk->stu", B, B)
    logBB = torch.log(BB)
    logPBB = logP[None] + logBB
    log1PBB = torch.log1p(-P[None] * BB)

    A = torch.einsum("sij,iu,suv,jv->", mD, Q, logPBB, Q)
    B = torch.einsum("sij,iu,suv,jv->", m1D, Q, log1PBB, Q)

    likelihood = 0.5 * (A + B)

    return -(prior + entropy + likelihood)


def adam_fit(labels, T, max_iter, rg):
    N, S = labels.shape
    K = labels.max() + 1

    masks, Delta = make_delta(labels)
    masks = torch.from_numpy(masks)
    Delta = torch.from_numpy(Delta)

    Q_logit = rg.normal(size=(N, T))
    P_logit = rg.normal(size=(T, T))
    P_logit = P_logit @ P_logit.T
    B_logit = rg.normal(size=(S, T, K))

    Q_logit = torch.tensor(Q_logit, requires_grad=True)
    P_logit = torch.tensor(P_logit, requires_grad=True)
    B_logit = torch.tensor(B_logit, requires_grad=True)

    opt = torch.optim.Adam([Q_logit, P_logit, B_logit], lr=1.0)
    elbos = []
    for i in range(max_iter):
        opt.zero_grad()
        # Q_logit.requires_grad_()
        # grad, loss = grad_and_value(adam_elbo)(Q_logit, masks, Delta)
        # Q_logit.grad = grad
        # Q_logit -= 0.01 * grad
        loss = elbo(Q_logit, P_logit, B_logit, masks, Delta)
        loss.backward()
        # if not i:
        #     print(torch.abs(Q_logit.grad).max())
        opt.step()
        elbos.append(loss.numpy(force=True))

        # if not i % 10:
        dq = torch.abs(Q_logit.grad).max()
        if dq < 1e-4:
            print(f"break {i=} {dq=}")
            break

        if i == 50:
            for g in opt.param_groups:
                g['lr'] = 0.1

    Q = torch.softmax(Q_logit, dim=1).numpy(force=True)
    P = torch.sigmoid(P_logit).numpy(force=True)
    pi = Q.mean(0)
    B = torch.softmax(B_logit, dim=2).numpy(force=True)
    return -np.array(elbos), Q, pi, P, B
