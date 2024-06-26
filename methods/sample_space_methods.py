import numpy as np

def build_space_env(K=3, links=np.array([-1, -0.5, 0, 0.5, 1])):
    links_indexed = np.arange(links.size)

    main_space = build_space(K, links)
    indexed_space = build_space(K, links_indexed).astype(int)
    matrix_space = build_space(K, links, as_matrix=True)

    return main_space, indexed_space, matrix_space


def build_space(K, links, as_matrix=False):

        a = links 
        c = len(links)
        s = K**2 - K

        S = np.zeros((c**s, s))

        for i in range(s):
            ou = np.tile(a, (int(c**(s-i-1)), 1)).flatten('F')
            os = tuple(ou for _ in range(c**i))
            o = np.concatenate(os)

            S[:, i] = o.T

        if not as_matrix:
            return S
        else:
            S_mat = np.zeros((c**s, K, K))

            for i in range(c**s):
                S_mat[i, :, :] = causality_matrix(S[i, :], fill_diag=1)
            
            return S_mat


def causality_matrix(link_vec, fill_diag=1):
    K = int(1/2 + np.sqrt(1-4*(-link_vec.size)) / 2)
    causal_mat = fill_diag * np.eye(K)
    causal_mat[~np.eye(K, dtype=bool)] = link_vec
    return causal_mat


def causality_vector(link_mat):
    return link_mat[~np.eye(link_mat.shape[0], dtype=bool)]


def construct_link_matrix(K):
    G = np.empty((K, K), dtype=object)
    for i in range(K):
        for j in range(K):
            if i == j:
                G[i, j] = ''
            else:
                G[i, j] = f'{i}->{j}'
    
    return G

