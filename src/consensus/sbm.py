import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, root
from sklearn.base import BaseEstimator, ClusterMixin
from torch.autograd.functional import jacobian
from tqdm.auto import trange


class SBMConsensus(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_blocks,
        em_max_iter=100,
        adam_max_iter=300,
        init_P_diag=0.9,
        init_P_offdiag=0.1,
        init_beta_dof=10,
        n_inits=50,
        random_state=None,
    ):
        self.n_blocks = n_blocks
        self.em_max_iter = em_max_iter
        self.adam_max_iter = adam_max_iter
        self.n_inits = n_inits
        self.init_P_diag = init_P_diag
        self.init_P_offdiag = init_P_offdiag
        self.init_beta_dof = init_beta_dof
        self.random_state = random_state

    def fit(self, X, y=None):
        best_elbo = -np.inf
        rg = np.random.default_rng(self.random_state)
        self.valid_ = np.any(X >= 0, axis=1)
        probs_ = None
        if self.valid_.any():
            # elbos, probs, pi, P = adam_fit(
            #     "smart", X[self.valid_], T=self.n_blocks, max_iter=self.max_iter
            # )
            # self.objectives_, probs, self.pi_, self.P_ = elbos, probs, pi, P
            # best_elbo = elbos[-1]
            # best_is_smart = True

            for f in trange(self.n_inits, desc="SBM inits"):
                em_good = True
                try:
                    elbos, probs, pi, P = daudin_fit(
                        X[self.valid_],
                        rg=rg,
                        max_iter=self.em_max_iter,
                        T=self.n_blocks,
                        init_P_diag=self.init_P_diag,
                        init_P_offdiag=self.init_P_offdiag,
                        init_beta_dof=self.init_beta_dof,
                    )
                except ValueError as e:
                    print(f"{f=} hit error {e=}")
                    em_good = False

                if em_good and elbos[-1] > best_elbo:
                    dprobs = None
                    if probs_ is not None:
                        dprobs = matched_probs_diff(probs, probs_)
                    self.objectives_, probs_, self.pi_, self.P_ = elbos, probs, pi, P
                    print("new em best", elbos[-1], "old best", best_elbo, "f", f)
                    if dprobs is not None:
                        print("em dprobs", dprobs)
                    best_elbo = elbos[-1]

                Q_init = rg.normal(size=(self.valid_.sum(), self.n_blocks))
                elbos, probs, pi, P = adam_fit(
                    Q_init,
                    X[self.valid_],
                    smart_correct=False,
                    T=self.n_blocks,
                    max_iter=self.adam_max_iter,
                )

                if elbos[-1] > best_elbo:
                    dprobs = None
                    if probs_ is not None:
                        dprobs = matched_probs_diff(probs, probs_)
                    self.objectives_, probs_, self.pi_, self.P_ = elbos, probs, pi, P
                    print("new adam best", elbos[-1], "old best", best_elbo, "f", f)
                    if dprobs is not None:
                        print("adam dprobs", dprobs)
                    best_elbo = elbos[-1]
                    # best_is_smart = False

                # elbos, probs, pi, P = adam_fit(
                #     Q_init, X[self.valid_], smart_correct=True, T=self.n_blocks, max_iter=self.max_iter
                # )
                # if elbos[-1] > best_elbo:
                #     self.objectives_, probs, self.pi_, self.P_ = elbos, probs, pi, P
                #     print("new smart best", elbos[-1], "old best", best_elbo)
                #     best_elbo = elbos[-1]
                #     best_is_smart = True
        # print(f"{best_is_smart=}")
        self.probabilities_ = np.zeros((X.shape[0], self.n_blocks))
        self.probabilities_[self.valid_] = probs_

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.valid_, self.probabilities_.argmax(1), -1)


def matched_probs_diff(q1, q2):
    scores = q1.T @ q2
    ii, jj = linear_sum_assignment(-scores)
    return np.abs(q1[:, ii] - q2[:, jj]).max(), np.abs(q1[:, ii] - q2[:, jj]).mean()


def beta_init_P(T, diag, offdiag, dof=None, random_state=None):
    eye = np.eye(T)
    if not dof:
        return diag * eye + offdiag * (1 - eye)

    rg = np.random.default_rng(random_state)
    P = rg.beta((1 - offdiag) * dof, offdiag * dof, size=(T, T))
    np.fill_diagonal(P, rg.beta((1 - diag) * dof, diag * dof, size=T))
    P = 0.5 * (P + P.T)
    return P


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


def e_step(Q_old, P, pi, masks, Delta):
    logP = torch.log(P)
    log1P = torch.log(1 - P)
    mD = masks * Delta
    m1D = masks * (1 - Delta)
    logpi = torch.log(pi)
    shape = Q_old.shape
    numel = Q_old.numel()
    # print(f" estep {P.min()=} {P.max()=}")
    # print(f" estep {logP.min()=} {logP.max()=}")
    # print(f" estep {log1P.min()=} {log1P.max()=}")

    def fun(QQ):
        # QQp = torch.softmax(QQ, dim=1)
        QQp = QQ
        A = torch.einsum("snj,tu,ju->nt", mD, logP, QQp)
        B = torch.einsum("snj,tu,ju->nt", m1D, log1P, QQp)
        QQQ = F.softmax(A + B + logpi, dim=1)
        return QQp - QQQ

    def funp(QQ):
        QQ = torch.from_numpy(QQ).view(shape)
        f = fun(QQ)
        return f.view(numel).numpy(force=True)

    def jac(QQ):
        j = jacobian(fun, QQ)
        return j

    def jacp(QQ):
        QQ = torch.from_numpy(QQ).view(shape)
        j = jac(QQ)
        return j.view(numel, numel).numpy(force=True)

    def fun_jac(QQ):
        QQ = torch.from_numpy(QQ).view(shape)
        f = fun(QQ)
        j = jac(QQ)
        print(".", end="")
        return f.view(numel).numpy(force=True), j.view(numel, numel).numpy(force=True)

    A = torch.einsum("snj,tu,ju->nt", mD, logP, Q_old)
    B = torch.einsum("snj,tu,ju->nt", m1D, log1P, Q_old)
    Qnew = F.softmax(A + B + logpi, dim=1)
    if torch.isclose(fun(Qnew), torch.tensor(0.0, dtype=torch.double)).all():
        Q = Qnew
    else:
        res = root(
            funp, x0=Qnew.numpy(force=True), method="krylov", options=dict(maxiter=100)
        )
        Q = torch.from_numpy(res.x).view(shape)
    # Q = torch.softmax(Q, dim=1)
    Q = torch.clip(Q, 0.0, 1.0)
    Q /= Q.sum(1)[:, None]

    # print(f" estep {Q.min()=} {Q.max()=}")
    # print(f" estep {Q.sum(1).min()=} {Q.sum(1).max()=}")
    good0 = torch.all(Q >= 0.0)
    good1 = torch.all(Q <= 1.0)
    good2 = torch.isclose(Q.sum(1), torch.tensor(1.0, dtype=torch.double)).all()

    if not (good0 and good1 and good2):
        raise ValueError(
            f"{good0=} {good1=} {good2=} {Q.min()=} {Q.max()=} {Q.sum(1).min()=} {Q.sum(1).max()=}"
        )

    return Q


def smart_init(T, masks, Delta, Q_old=None):
    P = torch.from_numpy(beta_init_P(T, 0.99, 0.1))
    pi = torch.ones(T, dtype=torch.double) / T
    if Q_old is None:
        Q_old = torch.ones((masks.shape[1], T), dtype=torch.double) / T
    else:
        Q_old = torch.as_tensor(Q_old)
    logP = torch.log(P)
    log1P = torch.log(1 - P)
    mD = masks * Delta
    m1D = masks * (1 - Delta)
    A = torch.einsum("snj,tu,ju->nt", mD, logP, Q_old)
    B = torch.einsum("snj,tu,ju->nt", m1D, log1P, Q_old)
    Q = torch.softmax(A + B + torch.log(pi), dim=1)

    numer = torch.einsum("sij,iu,jv->uv", masks * Delta, Q, Q)
    denom = torch.einsum("sij,iu,jv->uv", masks, Q, Q)
    # denom[denom == 0] = 1
    P = numer / denom
    # P = torch.nan_to_num(P)
    P = torch.clip(P, 1e-5, 1 - 1e-5)
    logP = torch.log(P)
    log1P = torch.log(1 - P)
    pi = Q.mean(0)
    A = torch.einsum("snj,tu,ju->nt", mD, logP, Q)
    B = torch.einsum("snj,tu,ju->nt", m1D, log1P, Q)
    Qnew = torch.log_softmax(A + B + torch.log(pi), dim=1)

    return Qnew


def update_pi(Q):
    pi = Q.mean(0)
    if torch.abs(pi.sum() - 1.0) > 1e-6:
        assert False, f"{torch.abs(pi.sum() - 1.0)=}"
    pi = torch.clip(pi, 1e-5, 1 - 1e-5)
    pi /= pi.sum()
    return pi


def update_P(Q, masks, Delta):
    # pi = pi / pi.sum()
    # print(f"{masks.shape=} {Delta.shape=} {masks.sum()=} {Delta.sum()=}")
    # stabilize = masks.shape[0]
    # denoms = [Q.T @ (masks[s]/masks[s].shape[1]) @ Q for s in range(len(masks))]  # torch.einsum("sij,iu,jv->uv", masks, Q, Q)
    # numers = [Q.T @ (masks[s] * Delta[s]/masks[s].shape[1]) @ Q for s in range(len(masks))]
    # Ps = [num / den for num, den in zip(numers, denoms)]
    # P = torch.mean(torch.stack(Ps, dim=0), dim=0)
    numer = torch.einsum("sij,iu,jv->uv", masks * Delta, Q, Q)
    denom = torch.einsum("sij,iu,jv->uv", masks, Q, Q)
    denom[denom == 0] = 1
    P = numer / denom
    P = torch.nan_to_num(P, posinf=0.0, neginf=0.0)
    P = torch.clip(P, 1e-5, 1 - 1e-5)
    assert torch.all(P >= 0)
    assert torch.all(P <= 1)
    return P


def elbo(Q, pi, P, masks, Delta):
    prior = (Q * torch.log(pi)).sum()
    # masked_Q = Q + (Q < 1e-8).to(torch.double)
    # entropy = (Q * torch.log(masked_Q)).sum()
    mQ = Q[Q > 1e-20]
    entropy = -(mQ * torch.log(mQ)).sum()

    mD = masks * Delta
    m1D = masks * (1 - Delta)
    logP = torch.log(P)  # + (P == 0).to(torch.double))
    log1P = torch.log(1 - P)  # + (P == 1).to(torch.double))
    # print(f" elbo {P.min()=} {P.max()=}")
    # print(f" elbo {logP.min()=} {logP.max()=}")
    # print(f" elbo {log1P.min()=} {log1P.max()=}")

    A = 0.5 * torch.einsum("sij,it,tu,ju->", mD, Q, logP, Q)
    B = 0.5 * torch.einsum("sij,it,tu,ju->", m1D, Q, log1P, Q)
    # print(f" elbo {prior=}")
    # print(f" elboz {entropy=}")
    # print(f" elbo {A=}")
    # print(f" elbo {B=}")
    likelihood = A + B
    return prior + entropy + likelihood


def daudin_fit(labels, rg, max_iter, T, init_P_diag, init_P_offdiag, init_beta_dof):
    P_init = beta_init_P(T, init_P_diag, init_P_offdiag, init_beta_dof, rg)

    N = labels.shape[0]
    masks, Delta = make_delta(labels)
    masks = torch.from_numpy(masks)
    Delta = torch.from_numpy(Delta)

    P = torch.from_numpy(P_init)
    pi = torch.ones(T, dtype=torch.double) / T
    Q = torch.ones((N, T), dtype=torch.double) / T
    # print(f" before {torch.min(P)=} {torch.max(P)=}")
    # print(f" before {torch.min(pi)=} {torch.max(pi)=}")
    # print(f" before {torch.min(Q)=} {torch.max(Q)=}")

    elbos = [elbo(Q, pi, P, masks, Delta).numpy(force=True)]
    for i in range(max_iter):
        Q_old = Q.clone()

        Q = e_step(Q, P, pi, masks, Delta)
        e0 = elbo(Q, pi, P, masks, Delta).numpy(force=True)
        # assert e0 + 1e-2 >= elbos[-1], f"0 {e0=} {elbos[-1]=}"
        if e0 + 1e-2 < elbos[-1]:
            raise ValueError(f"0 {i=} estep {e0=} {elbos[-1]=}")
        # print(f"--e {(e0>elbos[-1])=} {e0-elbos[-1]}")
        # print(f"   {i=} {torch.min(Q)=} {torch.max(Q)=}")
        # dQ = torch.max(torch.abs(Q_old - Q))
        # print(f"   {i=} {dQ=}")

        pi = update_pi(Q)
        e1 = elbo(Q, pi, P, masks, Delta).numpy(force=True)
        # assert e1 + 1e-2 >= e0, f"1 {i=} {e1=} {e0=} {elbos[-1]=}"
        if e1 + 1e-2 < e0:
            raise ValueError(f"1 {i=} {e1=} {e0=} {elbos[-1]=}")

        P = update_P(Q, masks, Delta)
        e2 = elbo(Q, pi, P, masks, Delta).numpy(force=True)
        # assert e2 + 1e-2 >= e1, f"2 {i=} {e2=} {e1=} {e0=} {elbos[-1]=}"
        if e2 + 1e-2 < e1:
            raise ValueError(f"2 {i=} {e2=} {e1=} {e0=} {elbos[-1]=}")
        # elbos.append(elbo(Q, pi, P, masks, Delta).numpy(force=True))

        # Q = e_step(Q, P, pi, masks, Delta)
        # e3 = elbo(Q, pi, P, masks, Delta).numpy(force=True)
        # print(f"--e {(e3>elbos[-1])=} {e3-elbos[-1]}")
        # print(f"--e {(e3>e2)=} {e3-e2}")
        # print(f"   {i=} {torch.min(Q)=}")
        # print(f"   {i=} {torch.max(Q)=}")
        dQ = torch.max(torch.abs(Q_old - Q))
        # print(f" -- {i=} {e2=} {e1=} {e0=} {elbos[-1]=}")
        # print(f" -- {i=} {dQ=}")
        # if dQ < 1e-6:
        #     break
        elbos.append(e2)
        if dQ < 1e-8:
            break

    Q = e_step(Q, P, pi, masks, Delta)
    elbos.append(elbo(Q, pi, P, masks, Delta).numpy(force=True))

    return elbos, Q.numpy(force=True), pi.numpy(force=True), P.numpy(force=True)


def adam_elbo(Q_logit, masks, Delta):
    Q = torch.softmax(Q_logit, dim=1)
    numer = torch.einsum("sij,iu,jv->uv", masks * Delta, Q, Q)
    denom = torch.einsum("sij,iu,jv->uv", masks, Q, Q)
    # denom[denom == 0] = 1
    P = numer / denom
    # P = torch.nan_to_num(P)
    P = torch.clip(P, 1e-5, 1 - 1e-5)
    pi = Q.mean(0)

    prior = (Q * torch.log(pi)).sum()
    # masked_Q = Q + (Q < 1e-8).to(torch.double)
    # entropy = (Q * torch.log(masked_Q)).sum()
    # mQ = Q[Q > 1e-8]
    # mQ = Q
    logQ = torch.log_softmax(Q_logit, dim=1)
    entropy = -(Q * logQ).sum()

    mD = masks * Delta
    m1D = masks * (1 - Delta)
    logP = torch.log(P)  # + (P == 0).to(torch.double))
    log1P = torch.log(1 - P)  # + (P == 1).to(torch.double))
    # print(f" elbo {P.min()=} {P.max()=}")
    # print(f" elbo {logP.min()=} {logP.max()=}")
    # print(f" elbo {log1P.min()=} {log1P.max()=}")

    A = 0.5 * torch.einsum("sij,it,tu,ju->", mD, Q, logP, Q)
    B = 0.5 * torch.einsum("sij,it,tu,ju->", m1D, Q, log1P, Q)
    # print(f" elbo {prior=}")
    # print(f" elboz {entropy=}")
    # print(f" elbo {A=}")
    # print(f" elbo {B=}")
    likelihood = A + B
    return -(prior + entropy + likelihood)


def adam_fit(Q_init, labels, T, smart_correct=False, max_iter=1000):
    # N = labels.shape[0]
    # T = P_init.shape[0]
    masks, Delta = make_delta(labels)
    masks = torch.from_numpy(masks)
    Delta = torch.from_numpy(Delta)
    if isinstance(Q_init, str) and Q_init == "smart":
        Q_logit = smart_init(T, masks, Delta)
        Q_logit = Q_logit.detach().clone()
    else:
        Q_logit = torch.from_numpy(Q_init)
        if smart_correct:
            Q_logit = smart_init(T, masks, Delta, Q_old=torch.softmax(Q_logit, dim=1))
            Q_logit = Q_logit.detach().clone()

    # P = torch.from_numpy(P_init)
    # pi = torch.ones(T, dtype=torch.double) / T
    # Q_logit = torch.log(torch.ones((N, T), dtype=torch.double) / T)
    # Q_logit = torch.normal(torch.zeros((N, T), dtype=torch.double), torch.ones((N, T), dtype=torch.double))
    Q_logit.requires_grad_()
    # Q_logit = torch.nn.Parameter(Q_logit, requires_grad=True)
    # Q_logit = torch.tensor(Q_logit, requires_grad=True)
    # Q_logit.requires_grad_()

    opt = torch.optim.Adam([Q_logit], lr=1.0)
    elbos = []
    for i in range(max_iter):
        opt.zero_grad()
        # Q_logit.requires_grad_()
        # grad, loss = grad_and_value(adam_elbo)(Q_logit, masks, Delta)
        # Q_logit.grad = grad
        # Q_logit -= 0.01 * grad
        loss = adam_elbo(Q_logit, masks, Delta)
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
                g["lr"] = 0.1

    Q = torch.softmax(Q_logit, dim=1)
    P = update_P(Q, masks, Delta)
    pi = update_pi(Q)
    return (
        -np.array(elbos),
        Q.numpy(force=True),
        pi.numpy(force=True),
        P.numpy(force=True),
    )
