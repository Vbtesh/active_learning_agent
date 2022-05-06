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
    num_var = int((1 + np.sqrt(1 + 4*len(link_vec))) / 2)
    causal_mat = fill_diag * np.ones((num_var, num_var))

    idx = 0
    for i in range(num_var):
        for j in range(num_var):
            if i != j:
                causal_mat[i, j] = link_vec[idx] 
                idx += 1
            
    return causal_mat


